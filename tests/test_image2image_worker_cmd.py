"""$image2image warm worker command selection."""

from __future__ import annotations

import unittest
from unittest import mock


class TestImage2ImageWorkerCmd(unittest.TestCase):
    def test_uses_anthill_media_venv(self) -> None:
        with mock.patch(
            "externals.comfy_inprocess.warm_worker.default_comfy_worker_cmd",
            return_value=["media-python", "-m", "externals.image2image.worker"],
        ) as mock_default:
            from externals.image2image.worker_cmd import build_image2image_worker_cmd

            cmd = build_image2image_worker_cmd()
        mock_default.assert_called_once()
        self.assertEqual(cmd[0], "media-python")
        self.assertIn("externals.image2image.worker", cmd[-1])


if __name__ == "__main__":
    unittest.main()
