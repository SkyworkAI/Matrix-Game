"""CUDA graphs around the existing online TAEHV, without changing time alignment.

The capture/state-restore implementation is covered by the project Apache-2.0
license. Network operations stay in the existing MIT TAEHV adapter.
Only batch1, one latent per chunk, inference on CUDA is supported.
"""
from types import SimpleNamespace

import torch


class StatefulGraph:
    def __init__(self, fn, example, memory):
        self.memory = {i: v.clone() for i, v in memory.items()}
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        self.fn = fn
        saved = {i: v.clone() for i, v in self.memory.items()}
        side = torch.cuda.Stream(device=example.device)
        current = torch.cuda.current_stream(example.device)
        side.wait_stream(current)
        try:
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn(self.input, self.memory)
            current.wait_stream(side)
            torch.cuda.synchronize(example.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.output = fn(self.input, self.memory)
        finally:
            for i, v in saved.items():
                self.memory[i].copy_(v)
        self.replays = 0

    def __call__(self, value, memory):
        if (value.shape != self.input.shape or value.dtype != self.input.dtype
                or value.device != self.input.device):
            raise ValueError("VAE graph signature changed")
        for i, persistent in self.memory.items():
            if memory[i] is not persistent:
                persistent.copy_(memory[i])
                memory[i] = persistent
        self.input.copy_(value)
        self.graph.replay()
        self.replays += 1
        return self.output.clone()


class StreamingVAEGraph(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.encoder_graph = self.decoder_graph = None

    def _apply(self, fn, *args, **kwargs):
        if self.encoder_graph is not None or self.decoder_graph is not None:
            torch.cuda.synchronize()
        self.encoder_graph = self.decoder_graph = None
        return super()._apply(fn, *args, **kwargs)

    def clear_cache(self):
        self.base.clear_cache()

    def encode(self, *args, **kwargs):
        return self.base.encode(*args, **kwargs)

    @torch.no_grad()
    def prime(self):
        if self.encoder_graph is not None and self.decoder_graph is not None:
            return
        p = next(self.parameters())
        try:
            self.clear_cache()
            for frames in (1, 4):
                latent = self.encode_stream(torch.zeros(1, 3, frames, 480, 864, device=p.device, dtype=p.dtype)).latent_dist.mode()
                self.decode_stream(latent)
            torch.cuda.synchronize(p.device)
        finally:
            self.clear_cache()

    def _check(self, value, channels, frames):
        if (torch.is_grad_enabled() or not value.is_cuda or value.ndim != 5
                or value.shape[:3] != (1, channels, frames)):
            raise ValueError("VAE graph requires CUDA inference, batch1/chunk1")

    @torch.no_grad()
    def encode_stream(self, pixels):
        self._check(pixels, 3, 1 if self.base._streaming_encode_frames == 0 else 4)
        if not self.base._streaming_encode_memory:
            return self.base.encode_stream(pixels)
        memory = self.base._streaming_encode_memory
        if self.encoder_graph is None:
            def encode(value, persistent):
                work = dict(persistent)
                video = value.permute(0, 2, 1, 3, 4).add(1).mul(0.5)
                out = self.base._encode_parallel_chunk(video, work).permute(0, 2, 1, 3, 4).contiguous()
                for i, tensor in persistent.items():
                    tensor.copy_(work[i])
                return out
            self.encoder_graph = StatefulGraph(encode, pixels, memory)
        latent = self.encoder_graph(pixels, memory)
        self.base._streaming_encode_frames += pixels.shape[2]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    @torch.no_grad()
    def decode_stream(self, latent):
        self._check(latent, 48, 1)
        state = self.base._streaming_decoder
        if state is None:
            return self.base.decode_stream(latent)
        if state.decoder_work_queue:
            raise ValueError("Decoder contains pending frames")
        memory = {i: v for i, v in enumerate(state.decoder_memory) if isinstance(v, torch.Tensor)}
        if self.decoder_graph is None:
            from matrix_game.inference.taehv import TWorkItem, apply_model_with_memblocks_sequential_single_step
            def decode(value, persistent):
                work_memory = [persistent.get(i) for i in range(len(self.base.core.decoder))]
                queue = [TWorkItem(value[:, :, 0], 0)]
                frames = []
                while queue:
                    frame = apply_model_with_memblocks_sequential_single_step(self.base.core.decoder, work_memory, queue)
                    if frame is not None:
                        frames.append(self.base.core.postprocess_output_frames(frame))
                for i, tensor in persistent.items():
                    tensor.copy_(work_memory[i])
                return torch.cat(frames, dim=1).mul(2).sub(1).permute(0, 2, 1, 3, 4).contiguous()
            self.decoder_graph = StatefulGraph(decode, latent, memory)
        pixels = self.decoder_graph(latent, memory)
        for i, value in memory.items():
            state.decoder_memory[i] = value
        state.n_frames_decoded += 4
        return SimpleNamespace(sample=pixels)
