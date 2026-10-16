#!/usr/bin/env python3
"""Reuse the shared T5 encoder and Wan2.2 VAE from the Hugging Face cache instead of downloading them again.

The release manifest (releases/models.json) lists the shared files base/models_t5_umt5-xxl-enc-bf16.pth and base/Wan2.2_VAE.pth with their
size and SHA256. The Wan-AI/Wan2.2-TI2V-5B repository, which other projects on this machine already downloaded, hosts files of the same
names. This script looks for them in the Hugging Face cache, and ONLY when the SHA256 equals the manifest's does it hard-link the file into
resources/base (same file system, no extra disk). A file that differs is never used, and a file already in resources/base is left alone.
Nothing is downloaded or deleted. Whatever remains missing is downloaded by `matrix-game download` as usual.

Usage:   python scripts/dgx/link_cached_base.py [--resources resources]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_REPOSITORIES = ["models--Wan-AI--Wan2.2-TI2V-5B", "models--alibaba-pai--Wan2.2-Fun-5B-Control"]
SHARED_NAMES = ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resources", type=Path, default=ROOT / "resources")
    args = ap.parse_args()

    manifest = json.loads((ROOT / "releases" / "models.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["bundles"]["shared"]}
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

    for name in SHARED_NAMES:
        entry = expected["base/" + name]
        destination = args.resources / "base" / name
        if destination.exists():
            print(f"{name}: already in {destination}, left alone")
            continue
        candidates = [path for repository in CACHE_REPOSITORIES for path in (hub / repository).glob(f"snapshots/*/{name}")]
        linked = False
        for candidate in candidates:
            source = candidate.resolve()
            if source.stat().st_size != entry["size"]:
                continue
            if sha256(source) != entry["sha256"]:
                print(f"{name}: {candidate} has the right size but a different SHA256, not used")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            print(f"{name}: SHA256 matches the manifest, hard-linked from {candidate}")
            linked = True
            break
        if not linked:
            print(f"{name}: no matching file in the Hugging Face cache, `matrix-game download` will fetch it")


if __name__ == "__main__":
    main()
