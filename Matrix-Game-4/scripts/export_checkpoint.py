"""Export only the EMA tensor dictionary; never enable arbitrary pickle loading."""
import argparse
import hashlib
import json
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default="generator_ema")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != ".safetensors":
        parser.error("Output must be a new .safetensors file")
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or args.key not in state:
        raise ValueError("Missing checkpoint key: " + args.key)
    state = state[args.key]
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Expected a nonempty tensor-only state dictionary")
    # Keep keys and dtypes exactly; the inference loader owns prefix handling.
    state = {k: v.detach().contiguous() for k, v in state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".safetensors.partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    save_file(state, str(temporary), metadata={"format": "pt", "checkpoint_key": args.key})
    with safe_open(str(temporary), framework="pt", device="cpu") as result:
        if set(result.keys()) != set(state):
            raise ValueError("Exported tensor keys differ")
        for key, value in state.items():
            if not torch.equal(result.get_tensor(key), value):
                raise ValueError("Exported tensor differs: " + key)
    h = hashlib.sha256()
    with temporary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    os.rename(temporary, args.output)
    report = dict(file=args.output.name, size=args.output.stat().st_size, sha256=h.hexdigest(),
                  key=args.key, tensors=len(state), parameters=sum(v.numel() for v in state.values()),
                  dtypes=sorted({str(v.dtype) for v in state.values()}), tensor_equality=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
