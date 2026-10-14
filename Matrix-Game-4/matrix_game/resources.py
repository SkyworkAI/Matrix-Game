"""Download or verify an explicit, versioned manifest; never extract archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative:
        raise ValueError("Invalid resource path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("Resource path must be relative without traversal")
    root = root.resolve()
    path = root.joinpath(*parts.parts)
    if root not in path.resolve().parents:
        raise ValueError("Resource escapes destination")
    return path


def files_for(manifest, scene):
    if manifest.get("schema") != 1 or scene not in ("castle", "ancient-egypt", "racing", "all"):
        raise ValueError("Unknown manifest schema or scene")
    bundles = manifest["bundles"]
    names = ["shared"] + (["castle", "ancient-egypt", "racing"] if scene == "all" else [scene])
    files, seen = [], set()
    for name in names:
        for entry in bundles.get(name, []):
            checked_path(Path("."), entry["path"])
            checked_path(Path("."), entry.get("repo_path", entry["path"]))
            if "repo_id" in entry:
                if (not isinstance(entry["repo_id"], str)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", entry["repo_id"])
                        or entry.get("repo_type") not in ("model", "dataset")
                        or not isinstance(entry.get("revision"), str)
                        or not re.fullmatch(r"[a-f0-9]{40}", entry["revision"])):
                    raise ValueError("External resources need a repository and immutable revision")
            elif "repo_type" in entry or "revision" in entry:
                raise ValueError("Repository overrides require repo_id")
            if entry["path"] in seen or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("Duplicate resource or invalid SHA256")
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError("Invalid resource length")
            seen.add(entry["path"])
            files.append(entry)
    if not files:
        raise ValueError("Manifest contains no resources for this scene")
    return files


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["size"] and digest(path) == entry["sha256"]


def install_file(source, destination, entry):
    if not matches(source, entry):
        raise ValueError("Downloaded resource checksum mismatch: " + entry["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
        if not matches(Path(temporary), entry):
            raise ValueError("Copied resource checksum mismatch")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", choices=("castle", "ancient-egypt", "racing", "all"), default="all")
    parser.add_argument("--resources", type=Path, default=Path("resources"))
    parser.add_argument("--repo-id")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="dataset")
    parser.add_argument("--revision", help="Immutable Hugging Face commit hash")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    files = files_for(manifest, args.scene)
    if args.action == "download":
        if not args.repo_id or not args.revision or not re.fullmatch(r"[a-f0-9]{40}", args.revision):
            parser.error("Download requires --repo-id and a full --revision commit hash")
        from huggingface_hub import hf_hub_download
    for entry in files:
        path = checked_path(args.resources, entry["path"])
        if not matches(path, entry):
            if args.action == "verify":
                raise ValueError("Missing or corrupted resource: " + entry["path"])
            source = Path(hf_hub_download(entry.get("repo_id", args.repo_id), entry.get("repo_path", entry["path"]),
                                         repo_type=entry.get("repo_type", args.repo_type),
                                         revision=entry.get("revision", args.revision)))
            install_file(source, path, entry)
        print("Verified", entry["path"])
