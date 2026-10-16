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


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.config_path = self.root / "releases/config.json"
        self.config = json.loads(self.config_path.read_text())
        self.models = json.loads((self.root / "releases/models.json").read_text())
        self.assets = json.loads((self.root / "releases/game-assets.json").read_text())

    def test_model_downloads_include_release_config_once(self):
        for scene in ("castle", "ancient-egypt", "racing", "all"):
            with self.subTest(scene=scene):
                entries = files_for(self.models, scene)
                configs = [e for e in entries if e.get("repo_path", e["path"]) == "config.json"]
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0]["path"], "release-config.json")
                self.assertNotIn("repo_id", configs[0])
                self.assertTrue(matches(self.config_path, configs[0]))
                self.assertNotIn("base/config.json", [e["path"] for e in entries])
                self.assertFalse(any(e.get("repo_path", e["path"]) == "config.json"
                                     for e in files_for(self.assets, scene)))

    def test_release_config_matches_manifest(self):
        self.assertEqual(self.config["schema"], 1)
        self.assertEqual(self.config["version"], self.models["version"])
        self.assertEqual(set(self.config["scenes"]), set(self.models["bundles"]) - {"shared"})
        for scene, paths in self.config["scenes"].items():
            self.assertEqual(set(paths), {"weights", "metadata", "inference", "sink_image"})
            self.assertEqual(set(paths.values()),
                             {e.get("repo_path", e["path"]) for e in self.models["bundles"][scene]})
        upstream = [e for e in self.models["bundles"]["shared"] if "repo_id" in e]
        self.assertTrue(upstream)
        for entry in upstream:
            self.assertEqual(entry["repo_id"], self.config["base_model"]["repo_id"])
            self.assertEqual(entry["revision"], self.config["base_model"]["revision"])

    def test_release_config_download_verify_and_reuse(self):
        entry = next(e for e in self.models["bundles"]["shared"] if e["path"] == "release-config.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "models.json"
            manifest.write_text(json.dumps(dict(schema=1, bundles={"shared": [entry]})))
            args = ["--manifest", str(manifest), "--scene", "castle",
                    "--resources", str(root / "resources"), "--repo-id", self.config["repo_id"],
                    "--repo-type", "model", "--revision", "a" * 40]
            download = Mock(return_value=str(self.config_path))
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
                main(["download"] + args)
                main(["verify"] + args)
                main(["download"] + args)
            download.assert_called_once_with(self.config["repo_id"], "config.json",
                                             repo_type="model", revision="a" * 40)
            self.assertTrue(matches(root / "resources/release-config.json", entry))
            self.assertFalse((root / "resources/base/config.json").exists())


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
