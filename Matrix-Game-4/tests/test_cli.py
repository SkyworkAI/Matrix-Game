import contextlib
import io
import unittest

from matrix_game.cli import main


class CLITests(unittest.TestCase):
    def test_subcommand_help(self):
        for name, flag in (("serve", "--reference-only"), ("render", "--take"), ("verify", "--manifest")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                main([name, "--help"])
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn(flag, output.getvalue())
