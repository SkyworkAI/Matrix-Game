"""Incremental Wan VAE decoding with persistent causal features."""
import torch

class IncrementalDecoder:
    """Wan2.2 VAE 逐潜变量增量解码。与 vae2_2.WanVAE_.decode 相同的计算（conv2 是 1x1x1、解码器按潜变量逐个
    过并带因果特征缓存），只是缓存跨调用保留，所以分几次推进去的结果与一次整段解码的前缀相同。不改 VAE 代码。"""

    def __init__(self, vae_wrapper):
        from matrix_game.inference.wan.modules.vae2_2 import count_conv3d
        self.vae = vae_wrapper
        self.m = vae_wrapper.model
        self.feat = [None] * count_conv3d(self.m.decoder)
        self.first = True

    @torch.no_grad()
    def push(self, z):
        """z：[1, n, 48, h, w] 归一化潜变量（与 decode_to_pixel 的输入相同）。返回新增像素帧 [T, 3, H, W]，[-1, 1]。"""
        from matrix_game.inference.wan.modules.vae2_2 import unpatchify
        p = next(self.m.parameters())
        scale = self.vae._scale(p.device, p.dtype)
        z = z[0].permute(1, 0, 2, 3).unsqueeze(0).to(device=p.device, dtype=p.dtype)      # [1, 48, n, h, w]
        z = z / scale[1].view(1, self.m.z_dim, 1, 1, 1) + scale[0].view(1, self.m.z_dim, 1, 1, 1)
        x = self.m.conv2(z)
        outs = []
        for i in range(x.shape[2]):
            idx = [0]
            if self.first:
                outs.append(self.m.decoder(x[:, :, i:i + 1], feat_cache=self.feat, feat_idx=idx, first_chunk=True))
                self.first = False
            else:
                outs.append(self.m.decoder(x[:, :, i:i + 1], feat_cache=self.feat, feat_idx=idx))
        out = unpatchify(torch.cat(outs, 2), patch_size=2)
        return out.float().clamp_(-1, 1).squeeze(0).permute(1, 0, 2, 3)                   # [T, 3, H, W]

