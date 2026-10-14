"""Display composition and API tests; no model or GPU required."""
import asyncio
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from matrix_game.server.app import Session, application, controls, display_frames, validate_renderer


class DisplayTests(unittest.TestCase):
    def test_software_renderer_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "software OpenGL"):
            validate_renderer({"renderer": "llvmpipe (LLVM)"})
        self.assertEqual(validate_renderer({"renderer": "NVIDIA A100"}), "NVIDIA A100")
        with self.assertLogs("castle", level="WARNING"):
            self.assertEqual(validate_renderer({"renderer": "llvmpipe"}, True), "llvmpipe")

    def test_mouse_reset(self):
        self.assertTrue(controls(dict(sequence=1, look_reset=True))["look_reset"])
        for value in (1, "true", None):
            with self.assertRaises(ValueError):
                controls(dict(sequence=1, look_reset=value))
    def test_same_frame_pairing(self):
        source = np.arange(4 * 12 * 20 * 3, dtype=np.uint8).reshape(4, 12, 20, 3)
        render = 255 - source
        self.assertIs(display_frames(source, render, False), render)
        pair = display_frames(source, render, True)
        self.assertEqual(pair.shape, (4, 12, 40, 3))
        np.testing.assert_array_equal(pair[:, :, :20], source)
        np.testing.assert_array_equal(pair[:, :, 20:], render)
        np.testing.assert_array_equal(source + render, np.full_like(source, 255))

    def test_reject_shape_mismatch(self):
        with self.assertRaises(ValueError):
            display_frames(np.zeros((1, 12, 20, 3)), np.zeros((4, 12, 20, 3)), True)


class SessionEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_defeat_is_normal_completion(self):
        async def stop(session):
            session.closed = True
        runtime = SimpleNamespace(config=lambda: {}, stop=stop)
        session = Session(runtime, 'model')
        session.game = SimpleNamespace(ended=dict(reason='defeated'))
        session.future = asyncio.get_running_loop().create_future()
        session.future.set_result(None)
        session.socket = SimpleNamespace(closed=False, send_json=AsyncMock())
        await session.watch()
        await asyncio.sleep(0)
        self.assertTrue(session.closed)
        self.assertIsNone(session.error)
        self.assertEqual(session.report['ended'], 'defeated')
        session.socket.send_json.assert_awaited_once_with(dict(type='ended', reason='defeated'))


class RequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.started = []

        async def start(mode, monitor=False):
            self.started.append((mode, monitor))
            return SimpleNamespace(id="test", monitor=monitor)

        runtime = self.runtime = SimpleNamespace(gen=object(), active=None, start=start,
                                  args=SimpleNamespace(fps=24, whep_port=8890))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        runtime.args.game_root = Path(self.temp.name)
        runtime.args.asset_root = runtime.args.game_root
        audio = runtime.args.game_root / "assets/combat-audio"
        audio.mkdir(parents=True)
        (audio / "0.wav").write_bytes(b"RIFF-audio-fixture")
        self.client = TestClient(TestServer(application(runtime)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_start_options(self):
        for body, expected in (({"mode": "model"}, False),
                               ({"mode": "model", "monitor": True}, True),
                               ({"mode": "reference"}, False)):
            response = await self.client.post("/api/start", json=body)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["monitor"], expected)
        self.assertEqual(self.started, [("model", False), ("model", True), ("reference", False)])

    async def test_invalid_options(self):
        for body in ({"mode": "reference", "monitor": True}, {"mode": "model", "monitor": "false"}):
            response = await self.client.post("/api/start", json=body)
            self.assertEqual(response.status, 400)
        self.assertFalse(self.started)

    async def test_stop_requires_json_and_clears_session(self):
        session = SimpleNamespace(id="test")
        self.runtime.active = session

        async def stop(active):
            self.assertIs(active, session)
            self.runtime.active = None

        self.runtime.stop = AsyncMock(side_effect=stop)
        response = await self.client.post("/api/test/stop")
        self.assertEqual(response.status, 415)
        self.runtime.stop.assert_not_awaited()
        response = await self.client.post("/api/test/stop", json={})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"stopped": True})
        self.assertIsNone(self.runtime.active)
        self.runtime.stop.assert_awaited_once_with(session)
        response = await self.client.post("/api/test/stop", json={})
        self.assertEqual(response.status, 410)

    async def test_private_service_boundary(self):
        response = await self.client.post("/api/start", data='{invalid', headers={"Content-Type": "application/json"})
        self.assertEqual(response.status, 400)
        response = await self.client.post("/api/start", data='{}', headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status, 415)
        response = await self.client.post("/api/start", json={}, headers={"Origin": "https://example.com"})
        self.assertEqual(response.status, 403)
        response = await self.client.get("/", headers={"Host": "example.com"})
        self.assertEqual(response.status, 403)
        for body in ([], None, {"command": "unexpected"}):
            response = await self.client.post("/api/start", data=__import__('json').dumps(body), headers={"Content-Type": "application/json"})
            self.assertEqual(response.status, 400)

    async def test_scene_title(self):
        for scene, title in (("castle", "Medieval Castle"), ("ancient-egypt", "Ancient Egypt")):
            self.runtime.args.scene = scene
            response = await self.client.get("/")
            text = await response.text()
            self.assertIn(f"<h1>{title}</h1>", text)
            self.assertNotIn("__SCENE_NAME__", text)

    async def test_audio_routes(self):
        response = await self.client.get("/audio.js")
        self.assertEqual(response.status, 200)
        self.assertIn("CastleAudio", await response.text())
        response = await self.client.get("/audio/0.wav")
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"RIFF-audio-fixture")
        for url in ("/audio/7.wav", "/audio/../serve_castle.py", "/audio/0.wav/extra"):
            self.assertEqual((await self.client.get(url)).status, 404)


if __name__ == "__main__":
    unittest.main()
