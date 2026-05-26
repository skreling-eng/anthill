"""Tests for $comfy running one API job per input image."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.comfy.run import run


def _write_png(path: Path) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT"
        b"\x08\xd7c\xf8\x0f\x00\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class TestComfyMultiImage(unittest.TestCase):
    def setUp(self) -> None:
        self._emulate = os.environ.pop("AH_EMULATE_COMFY", None)

    def tearDown(self) -> None:
        if self._emulate is not None:
            os.environ["AH_EMULATE_COMFY"] = self._emulate

    def test_emulate_one_job_per_image(self) -> None:
        os.environ["AH_EMULATE_COMFY"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        op_dir = session_dir / "1__comfy"
        op_dir.mkdir(parents=True, exist_ok=True)
        wf = op_dir / "wf.json"
        wf.write_text(
            '{"load": {"class_type": "LoadImage", "inputs": {"image": "INPUT_IMAGE"}}}',
            encoding="utf-8",
        )
        paths: list[str] = []
        for i in range(3):
            p = session_dir / f"in_{i}.png"
            _write_png(p)
            paths.append(str(p.relative_to(session_dir)).replace("\\", "/"))

        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        out = run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(images=paths),
                args={"json": str(wf)},
                prompt_text="",
            ),
        )
        self.assertEqual(len(out.texts), 3)
        self.assertTrue(all("emulated $comfy run" in ctx.read_link_text(t) for t in out.texts))

    @patch("externals.comfy.run.ComfyClient")
    def test_queues_once_per_image(self, mock_client_cls: MagicMock) -> None:
        session_dir = create_session_dir(Path("sessions"))
        op_dir = session_dir / "2__comfy"
        op_dir.mkdir(parents=True, exist_ok=True)
        wf = op_dir / "wf.json"
        wf.write_text(
            '{"load": {"class_type": "LoadImage", "inputs": {"image": "INPUT_IMAGE.png"}}}',
            encoding="utf-8",
        )
        paths: list[str] = []
        for i in range(2):
            p = session_dir / f"img_{i}.png"
            _write_png(p)
            paths.append(str(p.relative_to(session_dir)).replace("\\", "/"))

        client = MagicMock()
        client.upload_image.side_effect = ["up0.png", "up1.png"]
        client.queue_prompt.side_effect = ["pid0", "pid1"]
        client.wait_history.side_effect = [
            {"outputs": {}},
            {"outputs": {}},
        ]
        mock_client_cls.return_value = client

        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        with self.assertRaises(RuntimeError):
            run(
                ctx,
                ExternalInput(
                    bundle=ArrayBundle(images=paths),
                    args={"json": str(wf), "port": "8188"},
                    prompt_text="",
                ),
            )

        self.assertEqual(client.upload_image.call_count, 2)
        self.assertEqual(client.queue_prompt.call_count, 2)
        self.assertEqual(client.wait_history.call_count, 2)


if __name__ == "__main__":
    unittest.main()
