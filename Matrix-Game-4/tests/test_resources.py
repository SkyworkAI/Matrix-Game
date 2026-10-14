import argparse
import hashlib
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import json
from types import SimpleNamespace

from matrix_game.resources import checked_path, digest, files_for, install_file, matches, main
from matrix_game.scenes import add_model_arguments, resolve_model_arguments, parse_steps
from matrix_game.inference.offline import frame_blocks


class ResourceTests(unittest.TestCase):
    def test_path_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resources"
            root.mkdir()
            for bad in ("../escape", "/absolute", "C:/absolute", "a/../../b", "a\\b", "", "a/./b", "a//b", "a/"):
                with self.assertRaises(ValueError): checked_path(root, bad)
            (root / "link").symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaises(ValueError): checked_path(root, "link/escape")

    def test_verified_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/"source", Path(tmp)/"data/file"
            src.write_bytes(b"tensor-fixture")
            item = dict(path="data/file", size=src.stat().st_size, sha256=digest(src))
            install_file(src, dst, item)
            self.assertTrue(matches(dst, item))
            src.write_bytes(b"bad")
            with self.assertRaises(ValueError): install_file(src, dst, item)
            self.assertEqual(dst.read_bytes(), b"tensor-fixture")

    def test_manifest_selection(self):
        entry = lambda path: dict(path=path, size=0, sha256=hashlib.sha256(b"").hexdigest())
        manifest = dict(schema=1, bundles={"shared":[entry("common")], "castle":[entry("castle")], "ancient-egypt":[entry("egypt")]})
        self.assertEqual([f["path"] for f in files_for(manifest,"castle")], ["common","castle"])
        manifest["bundles"]["castle"].append(entry("common"))
        with self.assertRaises(ValueError): files_for(manifest,"castle")

    def test_repository_path_validation(self):
        for bad in ("../escape", "/absolute", "a\\b", "a//b", "", None):
            entry = dict(path="models/castle/model.safetensors", repo_path=bad,
                         size=0, sha256=hashlib.sha256(b"").hexdigest())
            with self.subTest(repo_path=bad), self.assertRaises(ValueError):
                files_for(dict(schema=1, bundles={"castle": [entry]}), "castle")

    def test_repository_path_download_keeps_local_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"tensor-fixture")
            entry = dict(path="models/castle/model.safetensors",
                         repo_path="fps-castle/model.safetensors",
                         size=fixture.stat().st_size, sha256=digest(fixture))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"castle": [entry]})))
            download = Mock(return_value=str(fixture))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                main(["download", "--manifest", str(manifest), "--scene", "castle",
                      "--resources", str(root / "resources"), "--repo-id", "org/model",
                      "--repo-type", "model", "--revision", "a" * 40])
            download.assert_called_once_with("org/model", entry["repo_path"], repo_type="model", revision="a" * 40)
            self.assertTrue(matches(root / "resources" / entry["path"], entry))
            self.assertFalse((root / "resources" / entry["repo_path"]).exists())

    def test_external_repository_requires_pin(self):
        base = dict(path="base/tokenizer.json", size=0, sha256=hashlib.sha256(b"").hexdigest())
        for override in (dict(repo_id="org/model"), dict(repo_id="org/model", repo_type="model", revision="main"),
                         dict(repo_id="../model", repo_type="model", revision="b" * 40),
                         dict(repo_id=None, repo_type="model", revision="b" * 40), dict(revision="b" * 40)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                files_for(dict(schema=1, bundles={"shared": [{**base, **override}]}), "castle")

    def test_external_repository_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"upstream-fixture")
            entry = dict(path="base/tokenizer.json", repo_path="tokenizer.json",
                         repo_id="upstream/base", repo_type="model", revision="b" * 40,
                         size=fixture.stat().st_size, sha256=digest(fixture))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"shared": [entry]})))
            download = Mock(return_value=str(fixture))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                main(["download", "--manifest", str(manifest), "--scene", "castle",
                      "--resources", str(root / "resources"), "--repo-id", "org/model",
                      "--repo-type", "model", "--revision", "a" * 40])
            download.assert_called_once_with("upstream/base", "tokenizer.json", repo_type="model", revision="b" * 40)
            self.assertTrue(matches(root / "resources" / entry["path"], entry))


class ConfigTests(unittest.TestCase):
    def config(self, *argv):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        return resolve_model_arguments(parser.parse_args(argv), require_model=False)

    def test_preset_contract(self):
        fast, quality = self.config(), self.config("--preset","quality")
        self.assertEqual((fast.steps,fast.first_steps,fast.vae_backend,fast.dit_runtime,fast.vae_graph), ((1000,),(1000,),"taehv","ring-fused",True))
        self.assertEqual((quality.steps,quality.first_steps,quality.vae_backend,quality.dit_runtime,quality.vae_graph), ((1000,500),(1000,500),"wan","native",False))
        self.assertIsNotNone(fast.sink_image)

    def test_override_and_invalid_config(self):
        args=self.config("--preset","quality","--steps","1000","--no-sink-image")
        self.assertEqual(args.first_steps,(1000,))
        self.assertIsNone(args.sink_image)
        with self.assertRaises(ValueError): self.config("--vae-backend","wan","--vae-graph")
        for bad in ("", "500", "1000,0", "1000,500,750", "1000,1000", "a"):
            with self.assertRaises(argparse.ArgumentTypeError):parse_steps(bad)

    def test_offline_blocks(self):
        for count in range(18):
            blocks=list(frame_blocks(iter(range(count))))
            self.assertEqual([v for b,n in blocks for v in b[:n]], list(range(count)))
            self.assertTrue(all(len(b)==(1 if i==0 else 4) for i,(b,n) in enumerate(blocks)))
            if count:
                b,n=blocks[-1]
                self.assertTrue(all(v==count-1 for v in b[n:]))
