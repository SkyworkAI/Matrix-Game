"""`FastStream` without host-device syncs before the frame download.

Inputs go through pinned memory, the scheduler tables stay on the device, and the upload and VAE
encoder run on a side stream, so the host queues the next chunk during the clean pass.
"""

import torch

from ..rtx5090.stream import FastStream as _BaseStream


class FastStream(_BaseStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timesteps = self.scheduler.timesteps.double().to(self.device)
        self._sigmas = self.scheduler.sigmas.double().to(self.device)
        self._side = torch.cuda.Stream(self.device)  # input upload and VAE encoder

    def _sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        # index_select, not `sigmas[index]`: indexing with a 0-dim CUDA tensor reads it back to the host.
        index = torch.argmin((self._timesteps - timestep.double()).abs()).reshape(1)
        sigma = self._sigmas.index_select(0, index).reshape(())
        return torch.where(timestep.double() == 0, torch.zeros_like(sigma), sigma)

    def _valid_pixels(self, pixels) -> bool:
        if not pixels.is_floating_point():
            return False
        # NaN propagates through aminmax and fails both comparisons.
        low, high = torch.aminmax(pixels)
        return bool(low >= -1) and bool(high <= 1)

    def _input_stream(self) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """(compute stream, stream for upload and encode). Profiling times every stage with events on
        the compute stream, so it keeps the inputs there."""
        current = torch.cuda.current_stream(self.device)
        return current, (current if self.profile else self._side)

    def _stage(self, value: torch.Tensor, current: torch.cuda.Stream) -> torch.Tensor:
        """Host or device tensor -> device copy queued on the active stream without blocking the host.

        A device tensor may still be written by `current`: the active stream waits for it and holds it.
        """
        if value.is_cuda:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            value.record_stream(stream)
            return value.to(self.device)
        staged = torch.empty_like(value, pin_memory=True)  # same strides: a straight memcpy
        staged.copy_(value)
        # The caching host allocator keeps `staged` alive until the copy has run.
        return staged.to(self.device, non_blocking=True)

    def _upload_pixels(self, pixels, param) -> torch.Tensor:
        # Stays on the input stream: `_encode` continues there.
        current, stream = self._input_stream()
        with torch.cuda.stream(stream):
            return self._stage(pixels, current).permute(1, 0, 2, 3).unsqueeze(0).to(param.device, param.dtype)

    def _encode(self, x, param):
        # Encode on the side stream, overlapping the previous chunk's clean pass.
        current, stream = self._input_stream()
        with torch.cuda.stream(stream):
            control = super()._encode(x, param)
        current.wait_stream(stream)
        control.record_stream(current)  # allocated on the input stream, used and freed on `current`
        return control

    def _upload_noise(self, noise) -> torch.Tensor:
        current, stream = self._input_stream()
        with torch.cuda.stream(stream):
            result = self._stage(noise, current)
        current.wait_stream(stream)
        result.record_stream(current)
        return result.to(self.dtype)


def warm_up(transformer, vae, prompt_embeds, sink_latent, sink_mode, sink_rope_mode, vae_backend, *, chunks: int = 23) -> None:
    """`rtx5090.stream.warm_up` on this stream class (compiles every kernel, captures every graph)."""
    stream = FastStream(transformer, vae, prompt_embeds, sink_latent=sink_latent, sink_mode=sink_mode,
                        sink_rope_mode=sink_rope_mode, vae_backend=vae_backend)
    try:
        for index in range(chunks):
            count = 1 if index == 0 else 4
            stream.push(torch.zeros(count, 3, 480, 864), frame_start=stream.frame)
    finally:
        stream.close()
