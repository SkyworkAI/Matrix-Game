import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("package_release", Path(__file__).parents[1] / "scripts/package_release.py")
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


class PackagingTests(unittest.TestCase):
    def test_ignore_generated_files_and_reject_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("LICENSE", "README.md", "matrix_game/ok.py", "game/build/generated.txt", ".DS_Store", "notes.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            self.assertEqual(set(packaging.source_files(root)), {Path(n) for n in ("LICENSE", "README.md", "matrix_game/ok.py")})
            (root / "matrix_game/weights.pt").write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                packaging.source_files(root)

    def test_reject_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").symlink_to("/etc/passwd")
            with self.assertRaises(ValueError):
                packaging.source_files(root)

    def test_only_public_docs_and_no_development_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            included = ("LICENSE", "README.md", "docs/online.md", "tests/test_example.py")
            excluded = ("docs/release-status.md", "docs/benchmark.md", "docs/private/online.md",
                        "game/test-results/report.json", "matrix_game/logs/debug.txt",
                        "tests/audits/notes.md", "game/example.egg-info/PKG-INFO")
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            self.assertEqual(set(packaging.source_files(root)), {Path(n) for n in included})

    def test_reject_credentials_and_third_party_secrets(self):
        cases = [
            ("configs/.env.local", "fixture"),
            ("game/private.key", "fixture"),
            ("matrix_game/.ssh/config", "fixture"),
            ("game/third_party/fixture.h", "hf_" + "A" * 32),
            ("configs/example.json", '"password": "' + "not-a-real-secret" + '"'),
            ("game/third_party/fixture.h", "/Users/" + "private/project/"),
            ("game/third_party/fixture.h", "192.168." + "20.30"),
        ]
        for name, content in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for required in ("README.md", "LICENSE"):
                    (root / required).write_text("fixture")
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                with self.assertRaises(ValueError):
                    packaging.source_files(root)

    def test_current_tree_scans(self):
        files = packaging.source_files(Path(__file__).parents[1])
        self.assertIn(Path("matrix_game/web/index.html"), files)
        self.assertNotIn(Path("game/build/CMakeCache.txt"), files)
