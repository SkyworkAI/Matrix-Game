"""Training-compatible RGB preprocessing."""
from __future__ import annotations

import numpy as np
from PIL import Image

def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Match the torchvision PIL bilinear resize and rounded center crop."""
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    size = (round(width * scale), round(height * scale))
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    left = round((size[0] - target_width) / 2)
    top = round((size[1] - target_height) / 2)
    return image.crop((left, top, left + target_width, top + target_height))


def frames_to_tensor(frames) -> torch.Tensor:
    """PIL 帧列表 -> float32 [F, 3, H, W]，x * 2/255 - 1。"""
    import torch
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])  # [F, H, W, 3]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    return t * (2.0 / 255.0) - 1.0


def tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    """[F, 3, H, W] in [-1, 1] -> uint8 [F, H, W, 3]（评测与写视频用）。"""
    import torch
    v = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return v.permute(0, 2, 3, 1).cpu().numpy()

