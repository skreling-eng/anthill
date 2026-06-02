"""Tests for tools/ffmpeg resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from externals.video_audio import ffmpeg_paths as fp


class TestFfmpegPaths(unittest.TestCase):
    def test_vendored_win64(self) -> None:
        with patch.object(fp, "_FFMPEG_ROOT", Path("unused")):
            tmp = Path(self.id().replace(".", "_").replace(":", "_"))
        # use real tmp
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ffmpeg" / "win64"
            root.mkdir(parents=True)
            (root / "ffmpeg.exe").write_bytes(b"x")
            (root / "ffprobe.exe").write_bytes(b"x")
            with patch.object(fp, "_FFMPEG_ROOT", Path(td) / "ffmpeg"), patch.object(
                fp, "platform_key", lambda: "win64"
            ), patch.object(sys, "platform", "win32"):
                self.assertTrue(fp.vendored_ready())
                self.assertTrue(fp.get_ffmpeg_exe().endswith("ffmpeg.exe"))

    def test_missing_raises(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with patch.object(fp, "_FFMPEG_ROOT", Path(td)), patch.object(
                fp, "platform_key", lambda: "win64"
            ), patch("shutil.which", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    fp.get_ffmpeg_exe()


if __name__ == "__main__":
    unittest.main()
