import glob
import inspect
import json
import math
import os
import types
from typing import List, Optional
import torch
from torch import nn

from matrix_game.inference.scheduler import SchedulerInterface, FlowMatchScheduler
from matrix_game.inference.wan.modules.tokenizers import HuggingfaceTokenizer
from matrix_game.inference.wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from matrix_game.inference.wan.modules.vae import _video_vae
from matrix_game.inference.wan.modules.vae2_2 import _video_vae as _video_vae22
from matrix_game.inference.wan.modules.t5 import umt5_xxl
from matrix_game.inference.wan.modules.causal_model import CausalWanModel


# ---------------------------------------------------------------------------
# Wan2.2 Fun-5B-Control 的默认路径。全部可由配置（构造参数）或环境变量覆盖。
#   CF_FUN5B_DIR      权重目录（DiT、VAE、umT5、google/ 分词器同目录）
#   CF_DIT_PATH       DiT：目录（含 config.json 与 diffusion_pytorch_model*.safetensors）或单个 safetensors
#   CF_T5_CHECKPOINT  umT5 编码器权重
#   CF_T5_TOKENIZER   umT5 分词器目录
#   CF_VAE22_PATH     Wan2.2 VAE 权重
# ---------------------------------------------------------------------------
FUN5B_DIR = os.environ.get("CF_FUN5B_DIR", "resources/base")
DEFAULT_DIT_PATH = os.environ.get("CF_DIT_PATH", FUN5B_DIR)
DEFAULT_T5_CHECKPOINT = os.environ.get("CF_T5_CHECKPOINT", os.path.join(FUN5B_DIR, "models_t5_umt5-xxl-enc-bf16.pth"))
DEFAULT_T5_TOKENIZER = os.environ.get("CF_T5_TOKENIZER", os.path.join(FUN5B_DIR, "google/umt5-xxl"))
DEFAULT_VAE22_PATH = os.environ.get("CF_VAE22_PATH", os.path.join(FUN5B_DIR, "Wan2.2_VAE.pth"))

# Fun-5B-Control 的 config.json（目录里没有 config.json 时用它，比如我们 SFT 存出来的单个 safetensors）
FUN5B_CONTROL_CONFIG = dict(
    model_type="ti2v", patch_size=(1, 2, 2), text_len=512, in_dim=148, dim=3072, ffn_dim=14336,
    freq_dim=256, text_dim=4096, out_dim=48, num_heads=24, num_layers=30, qk_norm=True,
    cross_attn_norm=True, eps=1e-6,
)
# 148 通道 = 带噪 48 + 控制 48 + 修补 48（恒零）+ 掩码 4（恒零），与 DiffSynth fun148 的 y 布局一致
CONTROL_CHANNELS = 48
# 权重里有、我们的模块里不建的键（Fun 的参考图分支，本项目不用）
IGNORED_KEY_PREFIXES = ("ref_conv.",)


def _to_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[str(dtype)]


def _rank0():
    import torch.distributed as dist
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _read_dit_config(model_path: str, config_path: Optional[str] = None) -> dict:
    """读 DiT 结构配置：显式 config_path 优先，其次 model_path（目录，或文件所在目录）下的 config.json，都没有就用 Fun-5B-Control 的配置。"""
    if config_path is None:
        d = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        cand = os.path.join(d, "config.json")
        config_path = cand if os.path.isfile(cand) else None
    if config_path is None:
        return dict(FUN5B_CONTROL_CONFIG)
    with open(config_path) as f:
        return json.load(f)


def _dit_weight_files(model_path: str, weights_path: Optional[str] = None) -> List[str]:
    if weights_path is not None:
        return [weights_path]
    if os.path.isfile(model_path):
        return [model_path]
    files = sorted(glob.glob(os.path.join(model_path, "diffusion_pytorch_model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{model_path} 下没有 diffusion_pytorch_model*.safetensors")
    return files


def _load_state_dict_files(files: List[str]) -> dict:
    sd = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(f, device="cpu"))
        else:
            obj = torch.load(f, map_location="cpu", weights_only=True)
            for k in ("generator_ema", "generator", "state_dict"):
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
            sd.update(obj)
    # 去掉整体前缀（DiffSynth 的 pipe.dit.、CF 包装器的 model.）
    for prefix in ("pipe.dit.", "dit.", "model."):
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def load_wan_dit(model_cls, model_path: str, config_path: Optional[str] = None,
                 weights_path: Optional[str] = None, dtype=torch.float32, verbose: bool = True,
                 load_weights: bool = True, **extra_init_kwargs):
    """按 config.json 建 WanModel / CausalWanModel 并载入权重。

    键名核对：除 IGNORED_KEY_PREFIXES（ref_conv.*）外，缺失或多余的键一律报错；忽略的键逐个打印。
    参数先建在 meta 设备上再 assign 载入，避免 5B 模型先随机初始化一遍。
    """
    cfg = _read_dit_config(model_path, config_path)
    sig = inspect.signature(model_cls.__init__).parameters
    init_kwargs = {k: v for k, v in cfg.items() if k in sig}
    dropped_cfg = sorted(k for k in cfg if k not in sig)
    if "patch_size" in init_kwargs:
        init_kwargs["patch_size"] = tuple(init_kwargs["patch_size"])
    # Wan2.2 官方 TI2V-5B 配置：cross_attn_norm=True、qk_norm=True（CF 两个类默认也是 True），显式写上防默认值被改
    init_kwargs.setdefault("cross_attn_norm", True)
    init_kwargs.setdefault("qk_norm", True)
    init_kwargs.update(extra_init_kwargs)

    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        model = model_cls(**init_kwargs)
    # RoPE 频率表不是参数也不是 buffer；调用方若在 torch.device("meta") 下构造，它会落在 meta 上，这里在 CPU 上重算
    if hasattr(model, "freqs") and model.freqs.device.type == "meta":
        from matrix_game.inference.wan.modules.model import rope_params
        d = model.dim // model.num_heads
        with torch.device("cpu"):
            model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                                     rope_params(1024, 2 * (d // 6))], dim=1)
    if not load_weights:
        # 只建结构（参数在 meta 上），供「rank0 读权重、其余 rank 由 FSDP sync_module_states 广播」
        if verbose and _rank0():
            print(f"[load_wan_dit] {model_cls.__name__} 只建结构不读权重（meta），config 来自 {model_path}", flush=True)
        return nn.Module.to(model, dtype=_to_dtype(dtype))

    files = _dit_weight_files(model_path, weights_path)
    sd = _load_state_dict_files(files)
    ignored = sorted(k for k in sd if k.startswith(IGNORED_KEY_PREFIXES))
    for k in ignored:
        sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"DiT 权重与结构不匹配：missing {len(missing)} {list(missing)[:5]}，"
                           f"unexpected {len(unexpected)} {list(unexpected)[:5]}")
    model = nn.Module.to(model, dtype=_to_dtype(dtype))  # 绕开 diffusers ModelMixin.to 的提示
    # freqs 不是 buffer（官方刻意如此，防止 .to(dtype) 改掉 complex 精度），不受上面 to 影响
    if verbose and _rank0():
        print(f"[load_wan_dit] {model_cls.__name__} <- {files}；config 键未用：{dropped_cfg}；"
              f"载入 {len(sd)} 个张量，忽略 {len(ignored)} 个（{ignored}，Fun 的参考图分支，本项目不用）；"
              f"missing 0、unexpected 0；patch_embedding {tuple(model.patch_embedding.weight.shape)}；dtype {model.patch_embedding.weight.dtype}",
              flush=True)
    return model


def init_kv_cache(generator, batch_size: int, kv_cache_size: int, dtype, device) -> List[dict]:
    """按模型实际层数、头数、头维建自注意力 KV 缓存（原来写死 30 层、[B, L, 12, 128]）。"""
    m = generator.model if hasattr(generator, "model") else generator
    n_head, head_dim = m.num_heads, m.dim // m.num_heads
    return [{
        "k": torch.zeros([batch_size, kv_cache_size, n_head, head_dim], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, kv_cache_size, n_head, head_dim], dtype=dtype, device=device),
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
    } for _ in range(m.num_layers)]


def init_crossattn_cache(generator, batch_size: int, dtype, device) -> List[dict]:
    """交叉注意力缓存，[B, text_len, 头数, 头维]（原来写死 [B, 512, 12, 128]）。"""
    m = generator.model if hasattr(generator, "model") else generator
    n_head, head_dim = m.num_heads, m.dim // m.num_heads
    return [{
        "k": torch.zeros([batch_size, m.text_len, n_head, head_dim], dtype=dtype, device=device),
        "v": torch.zeros([batch_size, m.text_len, n_head, head_dim], dtype=dtype, device=device),
        "is_init": False,
    } for _ in range(m.num_layers)]


def frame_seq_length_of(generator, latent_h: int, latent_w: int) -> int:
    """每个潜变量帧的 token 数 = (h/patch_h)*(w/patch_w)。480x864 -> 30x54 潜变量 -> 405；原来写死 1560。"""
    m = generator.model if hasattr(generator, "model") else generator
    return (latent_h // m.patch_size[1]) * (latent_w // m.patch_size[2])


class WanTextEncoder(torch.nn.Module):
    def __init__(self, t5_checkpoint_path: Optional[str] = None, tokenizer_path: Optional[str] = None) -> None:
        super().__init__()
        # 路径做成配置项（原来写死 wan_models/Wan2.1-T2V-1.3B/）。Wan2.1 与 Wan2.2-5B 是同一个 umT5-XXL。
        t5_checkpoint_path = t5_checkpoint_path or DEFAULT_T5_CHECKPOINT
        tokenizer_path = tokenizer_path or DEFAULT_T5_TOKENIZER

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(t5_checkpoint_path,
                       map_location='cpu', weights_only=True)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU（无 GPU 时退回参数所在设备，便于 CPU 单元测试）
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


# Wan2.2 VAE 的逐通道归一化统计，原样取自官方 wan/modules/vae2_2.py 的 Wan2_2_VAE（与 DiffSynth WanVideoVAE38 相同）
WAN22_VAE_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
WAN22_VAE_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class WanVAE22Wrapper(torch.nn.Module):
    """Wan2.2 VAE（48 通道，时间 4 倍、空间 16 倍，首帧单独成一个潜变量）。

    encode_to_latent(video [B, F_rgb, 3, H, W]，[-1, 1]) -> [B, 1+(F_rgb-1)//4, 48, H/16, W/16]
    decode_to_pixel(latent [B, F, 48, h, w]) -> [B, 1+4(F-1), 3, 16h, 16w]，[-1, 1]
    计算精度跟随本模块参数（.to(torch.bfloat16) 即 bf16 编码，与 DiffSynth 训练口径一致）；输出 float32。
    归一化 (mu - mean) / std 在计算精度下做，与 DiffSynth WanVideoVAE38 相同。
    """
    z_dim = 48
    temporal_stride = 4
    spatial_stride = 16

    def __init__(self, vae_path: Optional[str] = None):
        super().__init__()
        self.mean = torch.tensor(WAN22_VAE_MEAN, dtype=torch.float32)
        self.std = torch.tensor(WAN22_VAE_STD, dtype=torch.float32)
        self.model = _video_vae22(
            pretrained_path=vae_path or DEFAULT_VAE22_PATH,
            z_dim=48, dim=160, dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False)

    def _scale(self, device, dtype):
        return [self.mean.to(device=device).to(dtype), (1.0 / self.std).to(device=device).to(dtype)]

    @torch.no_grad()
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [B, F_rgb, 3, H, W] -> 模型要 [1, 3, F_rgb, H, W]，逐样本编码（官方同样逐样本）
        p = next(self.model.parameters())
        scale = self._scale(p.device, p.dtype)
        out = [self.model.encode(u.permute(1, 0, 2, 3).unsqueeze(0).to(device=p.device, dtype=p.dtype), scale).float().squeeze(0)
               for u in pixel]
        return torch.stack(out, dim=0).permute(0, 2, 1, 3, 4)  # [B, C, F, h, w] -> [B, F, C, h, w]

    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        assert not use_cache, "Wan2.2 VAE 包装暂不支持逐块缓存解码"
        p = next(self.model.parameters())
        scale = self._scale(p.device, p.dtype)
        out = [self.model.decode(u.permute(1, 0, 2, 3).unsqueeze(0).to(device=p.device, dtype=p.dtype), scale)
               .float().clamp_(-1, 1).squeeze(0) for u in latent]
        return torch.stack(out, dim=0).permute(0, 2, 1, 3, 4)  # [B, 3, F, H, W] -> [B, F, 3, H, W]


def _cfg_dict(args, key):
    v = getattr(args, key, None) if args is not None else None
    return dict(v) if v else {}


def build_vae_wrapper(args=None):
    """按配置选 VAE。配置键：vae_type（默认 wan22；wan21 为原 Wan2.1 16 通道）、
    vae_kwargs（原样传给构造函数，wan22 只有 vae_path 一个键）。"""
    vae_type = getattr(args, "vae_type", "wan22") if args is not None else "wan22"
    if vae_type == "wan22":
        return WanVAE22Wrapper(**_cfg_dict(args, "vae_kwargs"))
    if vae_type == "wan21":
        return WanVAEWrapper()
    raise ValueError(f"未知 vae_type: {vae_type}")


def build_text_encoder(args=None):
    """配置键 text_encoder_kwargs（t5_checkpoint_path、tokenizer_path），缺省用 Fun-5B-Control 目录。"""
    return WanTextEncoder(**_cfg_dict(args, "text_encoder_kwargs"))


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name=DEFAULT_DIT_PATH,
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            config_path=None,
            weights_path=None,
            model_dtype="float32",
            load_weights=True,
    ):
        """model_name 是存在的路径（目录或 safetensors 文件）时走 load_wan_dit（Wan2.2 Fun-5B-Control 等）；
        否则沿用原来的 wan_models/{model_name}/ from_pretrained（Wan2.1）。
        config_path：结构配置 json（缺省找同目录 config.json，再缺省用 Fun-5B-Control 配置）。
        weights_path：另给权重（比如我们 DiffSynth SFT 存的 step-N.safetensors），结构仍按 config。
        model_dtype：载入后参数精度，训练默认 float32（FSDP 混合精度另管），评测可用 bfloat16。
        load_weights=False：只在 meta 上建结构不读盘（丁的请求 1，配合 FSDP sync_module_states 从 rank0 广播）。
        """
        super().__init__()

        if os.path.exists(str(model_name)) or weights_path is not None:
            if is_causal:
                self.model = load_wan_dit(CausalWanModel, model_name, config_path, weights_path, model_dtype,
                                          load_weights=load_weights,
                                          local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = load_wan_dit(WanModel, model_name, config_path, weights_path, model_dtype,
                                          load_weights=load_weights)
        elif is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # 双向模型也支持逐帧时间步（首帧 t=0），两版都把 [B, F] 原样传进模型
        self.uniform_timestep = False
        self.is_causal = is_causal
        # 模型输入通道 = 输出（带噪）通道 + 条件通道（Fun-5B-Control: 148 = 48 + 100）
        self.extra_in_channels = self.model.in_dim - self.model.out_dim
        self.control_channels = min(CONTROL_CHANNELS, self.extra_in_channels)

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = None  # 原来写死 32760，现在每次前向按输入形状计算，见 _seq_len
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        # CF 的 _set_gradient_checkpointing(module, value) 与新版 diffusers 的签名不兼容，直接置位
        self.model.gradient_checkpointing = True

    def frame_seq_length(self, latent_h: int, latent_w: int) -> int:
        return frame_seq_length_of(self, latent_h, latent_w)

    def _seq_len(self, x_bfchw: torch.Tensor) -> int:
        return x_bfchw.shape[1] * self.frame_seq_length(x_bfchw.shape[-2], x_bfchw.shape[-1])

    def _with_condition(self, x_bfchw: torch.Tensor, control: Optional[torch.Tensor]) -> torch.Tensor:
        """[B, F, 48, h, w] 带噪潜变量 + 控制潜变量 -> 模型输入 [B, in_dim, F, h, w]。
        通道顺序：带噪 48 | 控制 48 | 零 52（修补 48 + 掩码 4）。control 为 None 时控制通道全零。"""
        x = x_bfchw.permute(0, 2, 1, 3, 4)
        if self.extra_in_channels <= 0:
            return x
        b, _, f, h, w = x.shape
        parts = [x]
        n_zero = self.extra_in_channels
        if control is not None:
            assert control.shape[0] == b and control.shape[1] == f and control.shape[2] == self.control_channels \
                and tuple(control.shape[-2:]) == (h, w), (control.shape, x_bfchw.shape)
            parts.append(control.permute(0, 2, 1, 3, 4).to(dtype=x.dtype, device=x.device))
            n_zero -= self.control_channels
        parts.append(x.new_zeros(b, n_zero, f, h, w))
        return torch.cat(parts, dim=1)

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        Convert the clean prediction into the flow parameterization.
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id]
        # t=0 的干净条件帧（首帧）sigma 取 0，pred_x0 就是输入本身（原实现取网格上最近的 sigma≈0.005）
        sigma_t = torch.where(timestep.to(sigma_t.device) == 0, torch.zeros_like(sigma_t), sigma_t).reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        
        classify_mode: Optional[bool] = False, # DF
        concat_time_embeddings: Optional[bool] = False, #DF
        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep
        
        cache_start: Optional[int] = None,
        drop_control: bool = False,
        tf_crop: bool = False,
        ctx: Optional[dict] = None,
        ctx_cache: Optional[List[dict]] = None,
        ctx_write: bool = False,
        ctx_control: Optional[torch.Tensor] = None,
        ctx_rope_pos: Optional[torch.Tensor] = None,
        rope_offset: int = 0,
        rope_start_frame: Optional[int] = None,
        sink_rope_mode: Optional[str] = None,
        kv_mem: Optional[List[dict]] = None,
        kv_capture: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """控制条件约定
        - conditional_dict["control_latent"]：[B, F, 48, h, w] 低模控制潜变量；缺省（或 drop_control=True）时控制通道全零。
          KV 缓存推理时可以给整段 F_total 帧，按 current_start 自动切出当前块对应的帧；也可以只给当前块。
        - 带噪流：通道拼成 [带噪 48 | 控制 48 | 零 52]。
        - 教师强制的干净流 clean_x：控制通道恒为零（干净历史不带控制）。
        - KV 缓存写入（用 t=0 的干净结果重跑那一次）：调用方传 drop_control=True，缓存里不带控制。
        timestep：[B, F]，逐帧；首帧作干净条件时调用方把 timestep[:, 0] 置 0、noisy[:, 0] 换成干净首帧潜变量。
        戊：tf_crop=True（教师强制、首帧为干净条件时）按 ZYT 4.1 裁掉 {c_{T-1}, n_0}；第 0 帧的 flow_pred 补零，
        因为首帧 t=0（sigma=0），pred_x0[:, 0] 仍等于输入首帧。要求 timestep[:, 0] 全为 0。
        上下文槽（全部缺省时与原实现逐位一致）：
        - 教师强制：ctx = {"latent" [B,K,48,h,w], "control" [B,K,48,h,w] 或 None, "valid" [B,K], "pos" [B,K], "rope_offset"}，
          上下文通道拼成 [上下文潜变量 48 | 上下文低模 48 | 零 52]（与窗口带噪流同一布局），序列 [上下文 K | 干净 | 带噪]。
        - KV 推理：ctx_cache 是每层的上下文槽；ctx_write=True 时 noisy_image_or_video 是上下文潜变量 [B,K,48,h,w]、
          ctx_control 是它们的低模潜变量、ctx_rope_pos 是 RoPE 帧位置，只写上下文槽（调用方给 timestep 全 0）；
          rope_offset 为全局帧号整体加的偏移，rope_start_frame 显式指定本次输入的 RoPE 帧号（换上下文后重算缓存用）。
        （只在 KV 推理时用，缺省 None 时调用与原来逐位相同）：sink_rope_mode（"fixed" 时 sink 按固定相对距离现加
        RoPE）、kv_mem（每层记忆槽，只读）、kv_capture（每层抓取槽，原地写不带 RoPE 的 k 与 v），含义见
        wan/modules/causal_model.py 的 CausalWanSelfAttention.forward。
        """
        prompt_embeds = conditional_dict["prompt_embeds"]
        if ctx_write:
            assert ctx_cache is not None and ctx_rope_pos is not None, "写上下文槽要给 ctx_cache 与 ctx_rope_pos"
            model_in = self._with_condition(noisy_image_or_video, ctx_control)
            flow_pred = self.model(
                model_in, t=timestep, context=prompt_embeds, seq_len=self._seq_len(noisy_image_or_video),
                kv_cache=None, crossattn_cache=crossattn_cache, ctx_cache=ctx_cache, ctx_write=True,
                ctx_rope_pos=ctx_rope_pos,
            ).permute(0, 2, 1, 3, 4)
            pred_x0 = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1)).unflatten(0, flow_pred.shape[:2])
            return flow_pred, pred_x0
        control = None if drop_control else conditional_dict.get("control_latent", None)
        if control is not None and control.shape[1] != noisy_image_or_video.shape[1]:
            # 整段控制，按当前块起点切
            assert kv_cache is not None and current_start is not None, \
                f"control_latent 帧数 {control.shape[1]} 与输入 {noisy_image_or_video.shape[1]} 不同，只在 KV 缓存推理时按 current_start 切片"
            fsl = self.frame_seq_length(noisy_image_or_video.shape[-2], noisy_image_or_video.shape[-1])
            start_frame = current_start // fsl
            control = control[:, start_frame:start_frame + noisy_image_or_video.shape[1]]

        input_timestep = timestep
        model_in = self._with_condition(noisy_image_or_video, control)
        seq_len = self._seq_len(noisy_image_or_video)

        logits = None
        # X0 prediction
        if kv_cache is not None:
            kv_kw = {}   # 上下文相关参数只在用到时传，缺省调用与原实现相同
            if ctx_cache is not None:
                kv_kw["ctx_cache"] = ctx_cache
            if rope_offset:
                kv_kw["rope_offset"] = int(rope_offset)
            if rope_start_frame is not None:
                kv_kw["rope_start_frame"] = int(rope_start_frame)
            if sink_rope_mode is not None and sink_rope_mode != "absolute":
                kv_kw["sink_rope_mode"] = str(sink_rope_mode)
            if kv_mem is not None:
                kv_kw["kv_mem"] = kv_mem
            if kv_capture is not None:
                kv_kw["kv_capture"] = kv_capture
            flow_pred = self.model(
                model_in,
                t=input_timestep, context=prompt_embeds,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                **kv_kw
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing：干净流不带控制
                tf_kw = {}
                if tf_crop:
                    assert bool((timestep[:, 0] == 0).all()), "tf_crop 只在首帧为 t=0 干净条件时使用"
                    tf_kw["tf_crop"] = True
                if ctx is not None:
                    # 上下文槽，通道布局与带噪流相同（上下文低模被丢弃时调用方给全零控制）
                    tf_kw["ctx_x"] = self._with_condition(ctx["latent"].to(noisy_image_or_video.dtype), ctx.get("control", None))
                    tf_kw["ctx_valid"] = ctx.get("valid", None)
                    tf_kw["ctx_pos"] = ctx["pos"]
                    tf_kw["rope_offset"] = int(ctx.get("rope_offset", 64))
                flow_pred = self.model(
                    model_in, # => [B, C_in, F, H, W]
                    t=input_timestep, context=prompt_embeds,
                    seq_len=seq_len,
                    clean_x=self._with_condition(clean_x, None), # => [B, C_in, F, H, W]
                    aug_t=aug_t,
                    **tf_kw,
                ).permute(0, 2, 1, 3, 4)
                if tf_crop:  # 戊：模型只返回第 1..F-1 帧，第 0 帧补零
                    flow_pred = torch.cat([torch.zeros_like(flow_pred[:, :1]), flow_pred], dim=1)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    flow_pred, logits = self.model(
                        model_in,
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                elif ctx is not None and not self.is_causal:
                    # 双向打分模型（real/fake score）带上下文（SFT 口径）：上下文通道 [上下文潜变量 48 | 上下文低模 48 |
                    # 零 52]（与教师强制上下文、窗口带噪流同一布局），拼在序列最前、t=0，时间位置 [上下文 pos | span..span+F)，
                    # 全双向注意力，输出头之前去掉上下文，输出只有窗口帧
                    ctx_x, ti = self._bidir_ctx_inputs(ctx, noisy_image_or_video)
                    flow_pred = self.model(
                        model_in,
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len + ctx_x.shape[2] * self.frame_seq_length(*noisy_image_or_video.shape[-2:]),
                        ctx_x=ctx_x, time_index=ti,
                    ).permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        model_in,
                        t=input_timestep, context=prompt_embeds,
                        seq_len=seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def _bidir_ctx_inputs(self, ctx: dict, noisy_image_or_video: torch.Tensor):
        """双向打分模型的上下文输入。ctx = {"latent" [B,K,48,h,w], "control" [B,K,48,h,w] 或 None,
        "valid" [B,K]（可缺省，有效的排前面）, "pos" [K] 或 [B,K]（context_time_indices 给的上下文 RoPE 位置）,
        "rope_offset"（窗口首帧位置，缺省 64）}。整批共用一组位置（频率表按批共用），所以各条样本的有效个数与位置要相同
        （DMD 部署口径 batch_size=1）。返回 (ctx_x [B, in_dim, n, h, w], time_index 长 n+F)。"""
        lat = ctx["latent"]
        B, K = lat.shape[:2]
        n = K
        if ctx.get("valid", None) is not None:
            nv = torch.as_tensor(ctx["valid"]).bool().reshape(B, K).long().sum(dim=1)
            assert bool((nv == nv[0]).all()), f"打分模型上下文：一批里各条有效个数要相同（频率表共用），收到 {nv.tolist()}"
            n = int(nv[0])
        pos = torch.as_tensor(ctx["pos"]).long()
        if pos.dim() == 2:
            assert bool((pos[:, :n] == pos[:1, :n]).all()), "打分模型上下文：一批里各条的位置要相同"
            pos = pos[0]
        pos = pos[:n]
        span = int(ctx.get("rope_offset", 64))
        F = noisy_image_or_video.shape[1]
        ctl = ctx.get("control", None)
        ctx_x = self._with_condition(lat[:, :n].to(noisy_image_or_video.dtype), None if ctl is None else ctl[:, :n])
        ti = [int(p) for p in pos.tolist()] + list(range(span, span + F))
        return ctx_x, ti

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
