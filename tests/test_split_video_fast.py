"""Tests for $split_video_fast."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.split_video_fast.run import run
from externals.split_video_fast.split import _scaled_dims
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestSplitVideoFastScaledDims(unittest.TestCase):
    def test_preserves_aspect_and_even_height(self) -> None:
        w, h = _scaled_dims(1920, 1080, 320)
        self.assertEqual(w, 320)
        self.assertEqual(h, 180)
        self.assertEqual(h % 2, 0)


class TestSplitVideoFastParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$split_video_fast(threshold=12, min_frames=80, width=320)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "split_video_fast")
        self.assertEqual(expr.args.get("threshold"), "12")
        self.assertEqual(expr.args.get("width"), "320")


class TestSplitVideoFastRun(unittest.TestCase):
    def test_emulate_outputs_labeled_fragments(self) -> None:
        os.environ["AH_EMULATE_SPLIT_VIDEO_FAST"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("split_video_fast")
            )
            bundle = ArrayBundle()
            bundle.videos.append(
                ctx.new_link("videos", ".mp4", b"\x00\x00\x00\x18ftypisom\x00")
            )
            inp = ExternalInput(
                bundle=bundle,
                args={"threshold": "10", "width": "320"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.labels), 1)
            self.assertEqual(out.labels[0][0], "fragment")
            self.assertEqual(out.labels[0][2]["frame"], 0)
            self.assertIn("src", out.labels[0][2])
        finally:
            os.environ.pop("AH_EMULATE_SPLIT_VIDEO_FAST", None)


if __name__ == "__main__":
    unittest.main()
