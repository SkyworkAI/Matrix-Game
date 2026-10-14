#!/usr/bin/env python3
#
# Vendored from https://github.com/madebyollin/taehv
# Upstream commit: 011dfc2112197741c540e0bdd5b7b67bcc930771
# License: MIT (see the upstream repository).  The original TAEHV network is
# kept in this file so the checkpoint and its kernels can be optimized locally.
"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class SuperMemBlock(nn.Module):
    """MemBlock variant used by the Super decoder (ConvNeXt-style: 7x7 depthwise conv + inverted bottleneck)."""
    def __init__(self, n_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f*2, n_f*2, 7, padding=3, groups=n_f*2, bias=False),
            nn.Conv2d(n_f*2, n_f*4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_f*4, n_f, 1, bias=False),
        )
    def forward(self, x, past):
        return self.conv(torch.cat([x, past], 1)) + x

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with parallelization over the time axis and iteration over blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N*T, C, H, W)

    # parallel over input timesteps, iterate over blocks
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, (MemBlock, SuperMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)

def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, (MemBlock, SuperMemBlock)):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i+1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N*b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i+1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT//b.stride, b.stride*C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i+1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i+1))
    return None

def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with iteration over timesteps as well as blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)

class _TAEHVCore(nn.Module):
    def __init__(self, checkpoint_path="taehv.pth", encoder_time_downscale=None, decoder_time_upscale=None, decoder_space_upscale=None, patch_size=None, latent_channels=None, arch_name=None):
        """Initialize pretrained TAEHV from the given checkpoint.

        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
            arch_name: checkpoint name to take the architecture from, like "taeh3" or "taehv1_5_super"
              (a "_super" suffix selects the larger, higher-quality decoder). Taken from checkpoint_path
              if None, so pass it explicitly if you've renamed the checkpoint file.
        Architecture arguments left as None are guessed from the checkpoint name.
        """
        super().__init__()
        # each architecture setting is guessed from the checkpoint name, unless it was passed explicitly
        self.arch_name = str(arch_name or checkpoint_path or "")
        if "taehv1_5" in self.arch_name: patch_size, latent_channels = patch_size or 2, latent_channels or 32
        if "taew2_2" in self.arch_name: patch_size, latent_channels = patch_size or 2, latent_channels or 48
        if "taeh3" in self.arch_name: patch_size, latent_channels = patch_size or 2, latent_channels or 24
        if "taeltx" in self.arch_name: # same for both 2 and 2.3
            patch_size, latent_channels = patch_size or 4, latent_channels or 128
            encoder_time_downscale, decoder_time_upscale = encoder_time_downscale or (True, True, True), decoder_time_upscale or (True, True, True)
        self.patch_size = patch_size or 1
        self.latent_channels = latent_channels or 16
        self.image_channels = 3
        encoder_time_downscale = encoder_time_downscale or (True, True, False)
        decoder_time_upscale = decoder_time_upscale or (False, True, True)
        decoder_space_upscale = decoder_space_upscale or (True, True, True)
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        self.encoder = nn.Sequential(
            conv(self.image_channels*self.patch_size**2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        if "_super" in self.arch_name:
            n_f = [512, 256, 128, 64]
            self.decoder = nn.Sequential(
                nn.Conv2d(self.latent_channels, n_f[0], 1, bias=False),
                SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), conv(n_f[0], n_f[1]*(2 if decoder_space_upscale[0] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[0] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
                SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), conv(n_f[1], n_f[2]*(2 if decoder_space_upscale[1] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[1] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
                SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), conv(n_f[2], n_f[3]*(2 if decoder_space_upscale[2] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[2] else 1), TGrow(n_f[3], 2 if decoder_time_upscale[2] else 1),
                conv(n_f[3], self.image_channels*self.patch_size**2),
            )
        else:
            n_f = [256, 128, 64, 64]
            self.decoder = nn.Sequential(
                Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
                MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
                MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
                MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
                nn.ReLU(inplace=True), conv(n_f[3], self.image_channels*self.patch_size**2),
            )
        # computed properties
        self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2**sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None:
            if str(checkpoint_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(str(checkpoint_path))
            else:
                sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(self.patch_tgrow_layers(sd))

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1: x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def _encode_h3_video(self, x, parallel, show_progress_bar):
        """Match H3's 17-frame chunks and three-token drop.
        https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/FL2VA/video_vae/klvae.py#L461-L503
        """
        batch = x.shape[0]
        x = torch.cat([x, x[:, -1:].expand(-1, -x.shape[1] % 17, -1, -1, -1)], dim=1)
        x = F.pad(x.reshape(batch, -1, 17, *x.shape[2:]), (0, 0, 0, 0, 0, 0, 3, 0))
        x = self.preprocess_input_frames(x)
        if parallel:
            x = apply_model_with_memblocks(self.encoder, x.flatten(0, 1), True, show_progress_bar)
            x = x.reshape(batch, -1, *x.shape[2:])
        else:
            x = torch.cat([apply_model_with_memblocks(self.encoder, chunk, False, False)
                           for chunk in tqdm(x.unbind(1), disable=not show_progress_bar)], dim=1)
        return x[:, :-3]

    def _decode_h3_video(self, x, parallel, show_progress_bar):
        """Match H3's five-token chunks and per-chunk prefix trim.
        https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/FL2VA/video_vae/klvae.py#L678-L786
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        chunk_frames = 5 * self.t_upscale
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % chunk_frames))
        x = x.unflatten(1, (-1, chunk_frames))[:, :, self.frames_to_trim:].flatten(1, 2)
        return self.postprocess_output_frames(x[:, :-3 * self.t_upscale])

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        if "taeh3" in self.arch_name:
            return self._encode_h3_video(x, parallel, show_progress_bar)
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1: x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=self.latent_channels) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        if "taeh3" in self.arch_name:
            return self._decode_h3_video(x, parallel, show_progress_bar)
        skip_trim = "taecvx" in self.arch_name and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        if skip_trim:
            # skip trimming for cogvideox to make frame counts match.
            # this still doesn't have correct temporal alignment for certain frame counts
            # (cogvideox seems to pad at the start?), but for multiple-of-4 it's fine.
            return x
        return x[:, self.frames_to_trim:]

class _StreamingTAEHV(nn.Module):
    def __init__(self, taehv):
        """Streaming wrapper around TAEHV for real-time use-cases (where not all inputs are available immediately).

        Encode-decode (video-to-video) usage:
            streaming = StreamingTAEHV(taehv)
            for frame in video_frames:
                latent = streaming.encode(frame_tensor)
                decoded = streaming.decode(latent)  # feeds latent if not None, then returns next frame
                if decoded is not None:
                    display(decoded)
            for frame in streaming.flush():
                display(frame)

        Decode-only (world model) usage:
            streaming = StreamingTAEHV(taehv)
            while running:
                latent = world_model.step()       # latent represents t_upscale frames
                frame = streaming.decode(latent)   # returns first frame immediately
                while frame is not None:           # retrieve remaining frames from this latent
                    display(frame)
                    frame = streaming.decode()
        """
        super().__init__()
        self.taehv = taehv
        self.reset()

    def reset(self):
        """Reset all internal state. Call this to start encoding/decoding a new stream."""
        self.encoder_work_queue, self.encoder_memory = [], [None] * len(self.taehv.encoder)
        self.decoder_work_queue, self.decoder_memory = [], [None] * len(self.taehv.decoder)
        self.n_frames_encoded, self.n_frames_decoded = 0, 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame (optional) and try to produce an encoder output.

        The encoder accumulates t_downscale input frames before producing one latent,
        so most calls will return None. Use flush_encoder() at end-of-stream to pad and
        drain any remaining latents.

        Args:
            x: NTCHW RGB frame tensor with values in [0, 1], or None to just process pending work.
        Returns: N1CHW latent tensor, or None if not enough input has been accumulated.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.image_channels, f"Expected NTCHW frames but got {x.shape=}"
            self._last_encoder_input_frame = x[:, -1:]
            x = self.taehv.preprocess_input_frames(x)
            self.encoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
            self.n_frames_encoded += x.shape[1]
        xt = apply_model_with_memblocks_sequential_single_step(
            self.taehv.encoder, self.encoder_memory, self.encoder_work_queue)
        return xt

    def decode(self, x=None):
        """Feed a latent (optional) and try to produce a decoded frame.

        Each latent produces t_upscale output frames due to temporal upscaling. The first
        decode(latent) call returns the first of these frames; call decode() with no argument
        to retrieve the rest, one at a time. Each call does the minimum decoder work needed to
        produce one frame.

        Startup frames (the first frames_to_trim raw decoder outputs, used for causal alignment
        with the reference VAE) are consumed internally and never returned.

        Args:
            x: NTCHW latent tensor, or None to retrieve the next pending frame.
        Returns: N1CHW decoded RGB frame tensor, or None if the queue needs more input.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.latent_channels, f"Expected NTCHW latents but got {x.shape=}"
            self.decoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.taehv.decoder, self.decoder_memory, self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # skip startup frames (to match decode_video trim behavior)
            if "taecvx" not in self.taehv.arch_name and self.n_frames_decoded <= self.taehv.frames_to_trim:
                continue
            return self.taehv.postprocess_output_frames(xt)

    def flush_encoder(self):
        """Pad (if needed) and drain all remaining latents from the encoder.

        Returns list of N1CHW latent tensors.
        """
        latents = []
        if self._last_encoder_input_frame is not None and self.n_frames_encoded % self.taehv.t_downscale != 0:
            n_pad = self.taehv.t_downscale - self.n_frames_encoded % self.taehv.t_downscale
            for _ in range(n_pad):
                lat = self.encode(self._last_encoder_input_frame)
                if lat is not None:
                    latents.append(lat)
        while (lat := self.encode()) is not None:
            latents.append(lat)
        return latents

    def flush_decoder(self):
        """Drain all remaining decoded frames from the decoder.

        Returns list of N1CHW decoded RGB frame tensors.
        """
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames

    def flush(self):
        """Flush encoder (with padding) and decoder, returning all remaining decoded frames.

        Returns list of N1CHW decoded RGB frame tensors.
        """
        frames = []
        for latent in self.flush_encoder():
            frame = self.decode(latent)
            if frame is not None:
                frames.append(frame)
        frames.extend(self.flush_decoder())
        return frames

@torch.no_grad()
def main():
    """Run TAEHV roundtrip reconstruction on the given video paths."""
    import os
    import sys
    import cv2 # no highly esteemed deed is commemorated here

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        def __iter__(self):
            return self
        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration  # End of video or error
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
        def __del__(self):
            if hasattr(self, 'writer'): self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    checkpoint_path = os.getenv("TAEHV_CHECKPOINT_PATH", "taehv.pth")
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    print(f"Using device \033[31m{dev}\033[0m, dtype \033[32m{dtype}\033[0m, checkpoint \033[34m{checkpoint_name}\033[0m ({checkpoint_path})")
    taehv = _TAEHVCore(checkpoint_path=checkpoint_path).to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        # convert to device tensor
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        video_out_path = video_path + f".reconstructed_by_{checkpoint_name}.mp4"
        video_out = VideoTensorWriter(video_out_path, (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        print(f"  Saved to {video_out_path}")

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Diffusers/Wan adapter
# ---------------------------------------------------------------------------
# The upstream implementation intentionally exposes NTCHW tensors in [0, 1].
# WanSelfForcingPipeline uses NCTHW tensors in [-1, 1] and expects the
# AutoencoderKL-style ``.latent_dist.sample()``/``.sample`` interface.  Keep
# those conversions here so callers do not need a second VAE code path.
from types import SimpleNamespace
from pathlib import Path
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class _DeterministicLatentDistribution:
    def __init__(self, value):
        self._value = value

    def sample(self, generator=None):
        del generator
        return self._value

    def mode(self):
        return self._value


class TAEHV(nn.Module):
    """Wan 2.2 5B TAEHV exposed through the pipeline's VAE interface.

    ``taew2_2`` is the only accepted architecture: it has patch size 2 and
    48 latent channels, matching Wan 2.2 5B.  In particular, this adapter
    refuses the 16-channel ``taehv``/Wan 2.1 checkpoint instead of silently
    producing incompatible latents.

    Encoding prefixes three first-frame copies to align the upstream pooling
    with Wan's causal 1+4k input boundaries. This is an approximate encoder;
    it does not reproduce the full Wan VAE's latent values exactly.
    """

    latent_channels = 48
    patch_size = 2
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    def __init__(self, checkpoint_path=None, *, arch_name=None, **kwargs):
        super().__init__()
        checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        name = str(arch_name or checkpoint_path or "").lower()
        if "taew2_2" not in name:
            raise ValueError(
                "TAEHV for Wan 2.2 5B requires a taew2_2 checkpoint "
                "(48 channels); refusing an incompatible 16-channel checkpoint"
            )
        self.core = _TAEHVCore(
            checkpoint_path=str(checkpoint_path),
            arch_name="taew2_2",
            **kwargs,
        )
        if self.core.latent_channels != self.latent_channels:
            raise ValueError(
                f"TAEHV checkpoint exposes {self.core.latent_channels} latent "
                f"channels, expected {self.latent_channels}"
            )
        self.config = SimpleNamespace(
            latent_channels=self.latent_channels,
            z_dim=self.latent_channels,
            spatial_compression_ratio=self.spatial_compression_ratio,
            temporal_compression_ratio=self.temporal_compression_ratio,
            scaling_factor=1.0,
            latents_mean=[0.0] * self.latent_channels,
            latents_std=[1.0] * self.latent_channels,
        )

    @property
    def dtype(self):
        """Track the actual module dtype after ``.to(dtype=...)``."""
        return next(self.parameters()).dtype

    @property
    def device(self):
        """Track the actual module device after ``.to(device=...)``."""
        return next(self.parameters()).device

    @staticmethod
    def _resolve_checkpoint(path):
        if path is None:
            raise ValueError("checkpoint_path is required for TAEHV")
        path = Path(path)
        if path.is_dir():
            candidates = sorted(path.glob("taew2_2*.safetensors"))
            candidates += sorted(path.glob("taew2_2*.pth"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "Expected exactly one taew2_2*.pth/.safetensors checkpoint "
                    f"under {path}, found {len(candidates)}"
                )
            path = candidates[0]
        if not path.is_file():
            raise FileNotFoundError(f"TAEHV checkpoint not found: {path}")
        return path

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """Diffusers-like loader accepting a checkpoint or model directory."""
        kwargs.pop("additional_kwargs", None)
        return cls(path, **kwargs)

    def clear_cache(self):
        """Reset active incremental encoder and decoder request state."""
        stream = getattr(self, "_streaming_encoder", None)
        del stream
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}
        self._streaming_encoder = None
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = None

    def begin_streaming_decode(self):
        """Start a causal decode session and return the adapter for chaining."""
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = _StreamingTAEHV(self.core)
        return self

    def begin_streaming_encode(self):
        """Start a causal encoder request without disturbing decoder state."""
        self._streaming_encoder = True
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}
        return self

    def end_streaming_encode(self):
        """Release causal encoder state at the end of a request."""
        self._streaming_encoder = None
        self._streaming_encode_frames = 0
        self._streaming_encode_batch = None
        self._streaming_encode_memory = {}

    def end_streaming_decode(self):
        """End a causal decode session, including partially accumulated state."""
        stream = getattr(self, "_streaming_decoder", None)
        if stream is not None:
            stream.reset()
        self._streaming_decoder = None

    @apply_forward_hook
    def encode_stream(self, x, return_dict=True, **kwargs):
        """Encode one causal pixel block and retain state across calls.

        The first call accepts ``4k+1`` frames (three prefix copies are added)
        and continuation calls accept ``4k`` frames.  The returned latent is
        NCTHW with three latent frames for the usual 9/12-pixel blocks.
        """
        del kwargs
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected NCTHW RGB input, got {tuple(x.shape)}")
        frames = x.shape[2]
        if not hasattr(self, "_streaming_encoder") or self._streaming_encoder is None:
            self.begin_streaming_encode()
        first = self._streaming_encode_frames == 0
        required_mod = 1 if first else 0
        if frames < 1 or frames % self.temporal_compression_ratio != required_mod:
            expected = "4k+1" if first else "4k"
            raise ValueError(f"streaming TAEHV encoder {expected} input required, got {frames} frames")
        if self._streaming_encode_batch is None:
            self._streaming_encode_batch = x.shape[0]
        elif self._streaming_encode_batch != x.shape[0]:
            raise ValueError(
                f"streaming TAEHV batch size changed from {self._streaming_encode_batch} to {x.shape[0]}"
            )
        video = x.permute(0, 2, 1, 3, 4).add(1).mul(0.5)
        if first:
            video = torch.cat((video[:, :1].expand(-1, 3, -1, -1, -1), video), dim=1)
        if video.shape[1] % self.temporal_compression_ratio:
            raise ValueError("streaming TAEHV encoder block must align temporal pooling")
        memory = self._streaming_encode_memory
        latent = self._encode_parallel_chunk(video, memory)
        self._streaming_encode_frames += frames
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        output = AutoencoderKLOutput(latent_dist=_DeterministicLatentDistribution(latent))
        return output if return_dict else (output.latent_dist,)

    def _encode_parallel_chunk(self, video, memory):
        batch = video.shape[0]
        hidden = self.core.preprocess_input_frames(video).flatten(0, 1)
        for index, layer in enumerate(self.core.encoder):
            if isinstance(layer, (MemBlock, SuperMemBlock)):
                shaped = hidden.reshape(batch, -1, *hidden.shape[1:])
                past = memory.get(index)
                if past is None:
                    past = torch.zeros_like(shaped[:, :1])
                history = torch.cat((past, shaped[:, :-1]), dim=1).reshape_as(hidden)
                memory[index] = shaped[:, -1:].clone()
                hidden = layer(hidden, history)
            else:
                hidden = layer(hidden)
        return hidden.reshape(batch, -1, *hidden.shape[1:])

    def _encode_chunked(self, x, chunk_frames=84):
        """Bound encoder activations while preserving memory across time chunks.

        Chunks include the initial three-frame prefix and end on complete
        temporal pooling groups. MemBlock history is its previous input, not
        its output; clone the last frame so it cannot retain a whole chunk.
        State is local to this call, including after an interrupted encode.
        """
        if chunk_frames < 4 or chunk_frames % self.temporal_compression_ratio:
            raise ValueError("TAEHV encoder chunks must be positive multiples of four")
        batch, _, frames, _, _ = x.shape
        memory = {}
        output = []
        start = 0
        while start < frames:
            prefix_frames = self.temporal_compression_ratio - 1 if start == 0 else 0
            end = min(start + chunk_frames - prefix_frames, frames)
            video = x[:, :, start:end].permute(0, 2, 1, 3, 4).add(1).mul_(0.5)
            if prefix_frames:
                video = torch.cat((video[:, :1].expand(-1, prefix_frames, -1, -1, -1), video), dim=1)
            output.append(self._encode_parallel_chunk(video, memory))
            start = end
        return torch.cat(output, dim=1)

    @apply_forward_hook
    def encode(self, x, return_dict=True, **kwargs):
        del kwargs
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"Expected NCTHW RGB input, got {tuple(x.shape)}")
        if (x.shape[2] - 1) % self.temporal_compression_ratio:
            raise ValueError("Causal Wan TAEHV encoding requires 4k+1 input frames")
        if x.shape[2] + self.temporal_compression_ratio - 1 > 84:
            latent = self._encode_chunked(x)
        else:
            # Wan/diffusers: NCTHW, [-1, 1] -> TAEHV: NTCHW, [0, 1].
            video = x.permute(0, 2, 1, 3, 4).add(1).mul_(0.5)
            # Upstream pools [0..3], [4..7], ... and pads the *end*. Wan's
            # block-causal contract instead uses [0], [1..4], [5..8], ... .
            # Prefix copies align those boundaries without reading future frames.
            # Keep this local adaptation outside the retained upstream network.
            prefix = video[:, :1].expand(-1, self.temporal_compression_ratio - 1, -1, -1, -1)
            video = torch.cat((prefix, video), dim=1)
            latent = self.core.encode_video(video, parallel=True, show_progress_bar=False)
        latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        output = AutoencoderKLOutput(
            latent_dist=_DeterministicLatentDistribution(latent)
        )
        return output if return_dict else (output.latent_dist,)

    @apply_forward_hook
    def decode(self, z, return_dict=True, **kwargs):
        del kwargs
        if z.ndim != 5 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected NCTHW latent with {self.latent_channels} channels, "
                f"got {tuple(z.shape)}"
            )
        latent = z.permute(0, 2, 1, 3, 4).contiguous()
        video = self.core.decode_video(latent, parallel=True, show_progress_bar=False)
        # TAEHV returns NTCHW [0, 1]; the pipeline's decode helper applies its
        # usual ``/ 2 + .5`` conversion after this method.
        video = video.mul(2).sub(1).permute(0, 2, 1, 3, 4).contiguous()
        output = DecoderOutput(sample=video)
        return output if return_dict else (output.sample,)

    @apply_forward_hook
    def decode_stream(self, z, return_dict=True, **kwargs):
        """Decode one latent block while retaining causal decoder state.

        The first block of F latent frames emits 4*F-3 pixel frames (nine for
        the normal three-latent block); continuation blocks emit 4*F pixel
        frames (twelve for three latents). The state
        is deliberately owned by this adapter so the pipeline can reset it at
        request boundaries and after failures.
        """
        del kwargs
        if z.ndim != 5 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected NCTHW latent with {self.latent_channels} channels, "
                f"got {tuple(z.shape)}"
            )
        if not hasattr(self, "_streaming_decoder") or self._streaming_decoder is None:
            self.begin_streaming_decode()
        latent = z.permute(0, 2, 1, 3, 4).contiguous()
        frames = []
        try:
            frame = self._streaming_decoder.decode(latent)
            while frame is not None:
                frames.append(frame)
                frame = self._streaming_decoder.decode()
        except Exception:
            self.end_streaming_decode()
            raise
        if frames:
            video = torch.cat(frames, dim=1)
        else:
            video = latent.new_empty((latent.shape[0], 0, 3, latent.shape[3] * self.spatial_compression_ratio, latent.shape[4] * self.spatial_compression_ratio))
        video = video.mul(2).sub(1).permute(0, 2, 1, 3, 4).contiguous()
        output = DecoderOutput(sample=video)
        return output if return_dict else (output.sample,)

    def forward(self, sample, sample_posterior=False, return_dict=True, **kwargs):
        del sample_posterior
        latent = self.encode(sample, return_dict=True, **kwargs).latent_dist.sample()
        return self.decode(latent, return_dict=return_dict)
