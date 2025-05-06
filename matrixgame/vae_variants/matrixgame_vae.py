from .wrapper import VAEWrapper
import os
import torch
from pathlib import Path
from matrixgame.vae_variants.matrixgame_vae_src import AutoencoderKLCausal3D

class MGVVAEWrapper(VAEWrapper):
    def __init__(self, vae):
        self.vae = vae
        self.vae.enable_tiling()
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode(self, x):
        x = self.vae.encode(x).latent_dist.sample()
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = x * self.vae.config.scaling_factor
        return x

    def decode(self, latents):
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample

def get_mg_vae_wrapper(model_path, weight_dtype):
    if model_path.endswith('.json'):
        model_path = os.splitext(model_path)[0]
    config = AutoencoderKLCausal3D.load_config(model_path)
    vae = AutoencoderKLCausal3D.from_config(config)
    vae_ckpt = Path(model_path) / "pytorch_model.pt"
    ckpt = torch.load(vae_ckpt)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    vae.load_state_dict(ckpt)
    vae.to(weight_dtype)
    return MGVVAEWrapper(vae)