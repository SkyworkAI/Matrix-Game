import os
import unittest
from unittest.mock import patch

from matrix_game.server.transport.output import sidecar_environment


class SidecarTests(unittest.TestCase):
    def test_model_python_paths_do_not_cross_interpreters(self):
        with patch.dict(os.environ, {"PYTHONPATH": "model-site", "PYTHONHOME": "model-home",
                                     "PYTHONUSERBASE": "model-user", "STUDIO_WEBRTC_RTSP_PORT": "8555"}):
            env = sidecar_environment()
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
                self.assertNotIn(key, env)
                self.assertIn(key, os.environ)
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["STUDIO_WEBRTC_RTSP_PORT"], "8555")
