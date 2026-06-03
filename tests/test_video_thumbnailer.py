"""Tests for $video_thumbnailer external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.video_thumbnailer.preview import (
    _grid_size,
    _human_duration,
    _human_size,
)
from externals.video_thumbnailer.run import run
from externals.video_thumbnailer.settings import ThumbnailOptions, options_from_input
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestParse(unittest.TestCase):
    def test_parse_args(self) -> None:
        expr = parse_actions("$video_thumbnailer(width=800, columns=3, rows=2)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "video_thumbnailer")
        self.assertEqual(expr.args.get("width"), "800")
        self.assertEqual(expr.args.get("columns"), "3")


class TestSettings(unittest.TestCase):
    def test_defaults(self) -> None:
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        opts = options_from_input(inp)
        self.assertEqual(opts.width, 1024)
        self.assertEqual(opts.columns, 5)
        self.assertEqual(opts.skip_seconds, 10.0)
        self.assertFalse(opts.no_header)

    def test_no_header_and_shadow_none(self) -> None:
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"no_header": "1", "timestamp_shadow_color": "none"},
            prompt_text="",
        )
        opts = options_from_input(inp)
        self.assertTrue(opts.no_header)
        self.assertIsNone(opts.timestamp_shadow_color)


class TestHelpers(unittest.TestCase):
    def test_human_size(self) -> None:
        self.assertIn("Ki", _human_size(2048))

    def test_human_duration(self) -> None:
        self.assertEqual(_human_duration(3661), "01:01:01")

    def test_vertical_grid_override(self) -> None:
        opts = ThumbnailOptions(
            columns=5,
            rows=5,
            vertical_video_columns=3,
            vertical_video_rows=4,
        )
        cols, rows = _grid_size(opts, aspect=0.5)
        self.assertEqual((cols, rows), (3, 4))


class TestEmulate(unittest.TestCase):
    def test_emulate_one_video(self) -> None:
        os.environ["AH_EMULATE_VIDEO_THUMBNAILER"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("vthumb"))
            bundle = ArrayBundle()
            bundle.videos.append(ctx.new_link("videos", ".mp4", b"fake"))
            inp = ExternalInput(bundle=bundle, args={"columns": "2"}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            self.assertEqual(len(out.videos), 0)
            data = (ctx.base_dir / out.images[0]).read_bytes()
            self.assertIn(b"[emulated $video_thumbnailer]", data)
        finally:
            os.environ.pop("AH_EMULATE_VIDEO_THUMBNAILER", None)

    def test_no_videos_raises(self) -> None:
        os.environ["AH_EMULATE_VIDEO_THUMBNAILER"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("vthumb"))
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            with self.assertRaises(RuntimeError):
                run(ctx, inp)
        finally:
            os.environ.pop("AH_EMULATE_VIDEO_THUMBNAILER", None)


if __name__ == "__main__":
    unittest.main()
