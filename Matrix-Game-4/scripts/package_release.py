#!/usr/bin/env python3
"""Build a new, allowlisted release directory without touching the working tree."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT_FILES = {"README.md", "LICENSE", "NOTICE", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
              "pyproject.toml", ".gitignore"}
ROOT_DIRS = {"matrix_game", "game", "configs", "docs", "licenses", "releases", "scripts", "tests"}
PUBLIC_DOCS = {"architecture.md", "installation.md", "offline.md", "online.md", "racing.md", "testing.md"}
SKIP = {".git", ".DS_Store", "__pycache__", ".pytest_cache", "node_modules", "build", "bin",
        "captures", "recordings", "offline-recordings", "runtime", "outputs", "resources",
        "audits", "wandb", "test-results", "playwright-report", "logs", "dist"}
SECRET_NAMES = {"id_rsa", "id_ed25519", "credentials", "known_hosts", ".netrc", ".pypirc"}
SECRET_DIRS = {".ssh", ".aws", ".codex", ".agents"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
PRIVATE = re.compile(
    r"/(?:Users|home|data|moe-xlc)/[A-Za-z0-9_.-]+/"
    r"|[A-Za-z]:\\Users\\[^\\\r\n]+\\|/root/(?:\.ssh|workspace|\.aws)/"
    r"|\b10\.(?:\d{1,3}\.){2}\d{1,3}\b"
    r"|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
    r"|-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    r"|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}"
    r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{25,}\b|\bxox[baprs]-[A-Za-z0-9-]{20,}\b"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bAIza[A-Za-z0-9_-]{30,}\b"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    r"|https?://[^\s/:@]+:[^\s/@]+@"
    r"|(?i:[?&](?:X-Amz-Signature|X-Goog-Signature|access_token|api_key|token)=[A-Za-z0-9_%+.-]{10,})"
    r"|(?i:[\"']?(?:api[_-]?key|access[_-]?token|client[_-]?secret|secret[_-]?key|password|authorization)"
    r"[\"']?\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"'])"
)


def source_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] not in ROOT_FILES | ROOT_DIRS or any(
            p in SKIP or p.startswith(".venv") or p.endswith(".egg-info") for p in rel.parts
        ):
            continue
        if rel.parts[0] == "docs" and (len(rel.parts) != 2 or rel.name not in PUBLIC_DOCS):
            continue
        if (set(rel.parts) & SECRET_DIRS or path.name in SECRET_NAMES
                or path.suffix.lower() in SECRET_SUFFIXES
                or path.name.startswith(".env") and path.name != ".env.example"):
            raise ValueError("Potential credential file in source: " + str(rel))
        if path.is_symlink():
            raise ValueError("Source symlink is not allowed: " + str(rel))
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".log", ".ipynb"}:
            continue
        if path.stat().st_size > 16 * 1024 * 1024 or path.suffix.lower() in {".pt", ".pth", ".safetensors", ".mp4", ".mkv", ".png", ".wav"}:
            raise ValueError("Unexpected large/binary source file: " + str(rel))
        # The vendored macOS app icon is binary; still scan its embedded strings.
        errors = "replace" if rel.as_posix() == "game/third_party/JoltPhysics/Build/macOS/icon.icns" else "strict"
        text = path.read_text(encoding="utf-8", errors=errors)
        if PRIVATE.search(text):
            raise ValueError("Potential private information in: " + str(rel))
        files.append(rel)
    if not {Path("LICENSE"), Path("README.md")} <= set(files):
        raise ValueError("Source tree must contain README and LICENSE")
    return files


def copy_source(root, output):
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the source tree")
    files = source_files(root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for rel in files:
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, destination)
        destination.chmod(0o755 if rel.suffix == ".sh" else 0o644)
        manifest.append(dict(path=rel.as_posix(), size=destination.stat().st_size,
                             sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def copy_resources(root, manifest_path, output):
    from matrix_game.resources import checked_path, files_for, matches
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the resource tree")
    manifest = json.loads(manifest_path.read_text())
    entries = files_for(manifest, "all")
    for entry in entries:
        if not matches(checked_path(root, entry["path"]), entry):
            raise ValueError("Missing or corrupted resource: " + entry["path"])
    for name in ("LICENSE", "README.md"):
        if not (root / name).is_file():
            raise ValueError("Resource distribution needs " + name)
    output.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        destination = checked_path(output, entry["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_path(root, entry["path"]), destination)
    for name in ("LICENSE", "README.md"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(manifest_path, output / "resource-manifest.json")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("source", "resources"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "resources" and args.manifest is None:
        parser.error("Resource packaging requires --manifest")
    root, output = args.root.resolve(), args.output.resolve()
    entries = (copy_source(root, output) if args.kind == "source" else
               copy_resources(root, args.manifest, output))
    print(json.dumps(dict(files=len(entries), bytes=sum(e["size"] for e in entries), output=str(output))))


if __name__ == "__main__":
    main()
