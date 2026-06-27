"""Tests for $avatar stage progress logging."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from externals.avatar.progress import (
    avatar_stage,
    log_avatar_stage_end,
    log_avatar_stage_start,
)


class TestAvatarProgress(unittest.TestCase):
    def test_avatar_stage_logs_start_and_done(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            with avatar_stage("test stage", detail="demo"):
                pass
        out = buf.getvalue()
        self.assertIn("$avatar: test stage (demo)…", out)
        self.assertIn("$avatar: test stage (demo) done (", out)

    def test_log_avatar_stage_start_end_pair(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            t0 = log_avatar_stage_start("WanVideoVAELoader")
            log_avatar_stage_end("WanVideoVAELoader", t0, note="ok")
        out = buf.getvalue()
        self.assertIn("$avatar: WanVideoVAELoader…", out)
        self.assertIn("$avatar: WanVideoVAELoader done (", out)
        self.assertIn(", ok)", out)


if __name__ == "__main__":
    unittest.main()
