"""$image2image warm worker command selection."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock


class TestImage2ImageWorkerCmd(unittest.TestCase):
    def tearDown(self) -> None:
        for key in ("AH_IMAGE2IMAGE_USE_MEDIA_VENV", "AH_COMFY_PYTHON"):
            os.environ.pop(key, None)

    def test_prefers_comfy_python_when_available(self) -> None:
        comfy = Path(r"G:\ComfyUI_V\.venv\Scripts\python.exe")
        with mock.patch(
            "externals.comfy_inprocess.bootstrap.resolve_comfy_python",
            return_value=comfy,
        ):
            from externals.image2image.worker_cmd import build_image2image_worker_cmd

            cmd = build_image2image_worker_cmd()
        self.assertEqual(cmd[0], str(comfy))
        self.assertIn("externals.image2image.worker", cmd[-1])

    def test_force_media_venv(self) -> None:
        os.environ["AH_IMAGE2IMAGE_USE_MEDIA_VENV"] = "1"
        with mock.patch(
            "externals.comfy_inprocess.bootstrap.resolve_comfy_python",
            return_value=Path(r"G:\ComfyUI_V\.venv\Scripts\python.exe"),
        ):
            with mock.patch(
                "externals.comfy_inprocess.warm_worker.default_comfy_worker_cmd",
                return_value=["media-python", "-m", "externals.image2image.worker"],
            ) as mock_default:
                from externals.image2image.worker_cmd import build_image2image_worker_cmd

                cmd = build_image2image_worker_cmd()
        mock_default.assert_called_once()
        self.assertEqual(cmd[0], "media-python")


if __name__ == "__main__":
    unittest.main()
