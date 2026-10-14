"""Strict inference checkpoint loading; no training dependencies."""
from typing import Optional, Sequence
import torch

_STRIP_INNER = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")


def clean_state_dict_keys(sd):
    if not isinstance(sd, dict) or not sd:
        raise ValueError("Checkpoint must contain a nonempty tensor dictionary")
    out = {}
    for k, v in sd.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            raise ValueError("Checkpoint contains non-tensor entries")
        for s in _STRIP_INNER:
            k = k.replace(s, "")
        if k in out:
            raise ValueError("Checkpoint key collision after prefix cleanup: " + k)
        out[k] = v
    for prefix in ("pipe.dit.", "dit.", "model."):
        if out and all(k.startswith(prefix) for k in out):
            out = {k[len(prefix):]: v for k, v in out.items()}
    return out


def load_dit_state_dict(path: str, keys: Sequence[str] = ("generator_ema", "generator")):
    """Read tensor-only safetensors or a restricted legacy state dictionary."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return clean_state_dict_keys(load_file(path, device="cpu")), "safetensors"
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    used = None
    if isinstance(obj, dict):
        for k in list(keys) + ["state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict) and len(obj[k]) > 0:
                obj, used = obj[k], k
                break
    return clean_state_dict_keys(obj), used


def build_dit(model_dir: str, ckpt: Optional[str], is_causal: bool, dtype="float32", timestep_shift=5.0,
              ckpt_keys: Sequence[str] = ("generator_ema", "generator"), load_weights=True, verbose=True, **kw):
    """Build on meta, then load exact weights, ignoring only unused ref_conv keys."""
    from matrix_game.inference.wrappers import WanDiffusionWrapper, IGNORED_KEY_PREFIXES, _to_dtype
    if not load_weights:
        return WanDiffusionWrapper(model_name=model_dir, is_causal=is_causal, timestep_shift=timestep_shift,
                                   model_dtype=dtype, load_weights=False, **kw)
    if ckpt is None:
        return WanDiffusionWrapper(model_name=model_dir, weights_path=ckpt, is_causal=is_causal,
                                   timestep_shift=timestep_shift, model_dtype=dtype, **kw)
    w = WanDiffusionWrapper(model_name=model_dir, is_causal=is_causal, timestep_shift=timestep_shift,
                            model_dtype=dtype, load_weights=False, **kw)
    sd, used = load_dit_state_dict(ckpt, ckpt_keys)
    ignored = [k for k in sd if k.startswith(IGNORED_KEY_PREFIXES)]
    for k in ignored:
        sd.pop(k)
    missing, unexpected = w.model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}[{used}] 与结构不匹配：missing {len(missing)} {list(missing)[:5]}，"
                           f"unexpected {len(unexpected)} {list(unexpected)[:5]}")
    w.model = torch.nn.Module.to(w.model, dtype=_to_dtype(dtype))
    w.model.eval()
    if verbose and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        print(f"[build_dit] {'causal' if is_causal else 'bidir'} <- {ckpt}[{used}]，{len(sd)} 个张量，dtype {dtype}", flush=True)
    return w

