"""Tests for $video_clip external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.video_clip.encode import max_segments
from externals.video_clip.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestVideoClipParse(unittest.TestCase):
    def test_parse_video_clip_args(self) -> None:
        expr = parse_actions("$video_clip(fps=25, delete_last_frames=3)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "video_clip")
        self.assertEqual(expr.args.get("fps"), "25")
        self.assertEqual(expr.args.get("delete_last_frames"), "3")


class TestVideoClipSegments(unittest.TestCase):
    def test_max_segments_matches_legacy_formula(self) -> None:
        # 10s audio, 81 frames/chunk at 25fps => 3.24s/segment => int(10/3.24)+1 = 4
        self.assertEqual(max_segments(10.0, frames_per_chunk=81, delete_last_frames=0, fps=25.0), 4)

    def test_max_segments_with_delete_last(self) -> None:
        # chunk = (81-10)/25 = 2.84s; 10/2.84+1 = 4
        self.assertEqual(max_segments(10.0, frames_per_chunk=81, delete_last_frames=10, fps=25.0), 4)


class TestVideoClipRuntime(unittest.TestCase):
    def test_emulated_run(self) -> None:
        os.environ["AH_EMULATE_VIDEO_CLIP"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("video_clip")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(
            videos=["videos/a.mp4", "videos/b.mp4"],
            sounds=["sounds/track.wav"],
        )
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        out = run(ctx, inp)
        self.assertEqual(len(out.videos), 1)
        self.assertEqual(out.sounds, ["sounds/track.wav"])
        self.assertTrue(out.videos[0].endswith(".mp4"))

    def test_requires_videos_and_sounds(self) -> None:
        os.environ["AH_EMULATE_VIDEO_CLIP"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("video_clip")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(videos=["v.mp4"]), args={}, prompt_text="")
        with self.assertRaises(RuntimeError):
            run(ctx, inp)

    def test_integration_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        os.environ["AH_EMULATE_VIDEO_CLIP"] = "1"
        os.environ["AH_EMULATE_FILE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,video_clip"
        source = """
@v: $file('clip.mp4')
@s: $file('track.wav')
@run: (@v, @s) -> $video_clip(fps=25)
"""
        result, _ = _run(source, "run")
        self.assertEqual(len(result.videos), 1)
        self.assertEqual(len(result.sounds), 1)


if __name__ == "__main__":
    unittest.main()
