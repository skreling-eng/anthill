"""Tests for $add_soft_subtitles."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.add_soft_subtitles.run import run as add_soft_subtitles_run
from externals.api import ExternalContext, ExternalInput
from externals.video_audio.ffmpeg_io import (
    is_ass_subtitle,
    is_subtitle_file,
    pair_videos_and_text_links,
    pair_videos_and_texts,
)
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$add_soft_subtitles(size=36)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "add_soft_subtitles")


class TestAssDetection(unittest.TestCase):
    def test_script_info(self) -> None:
        self.assertTrue(is_ass_subtitle("[Script Info]\nTitle: x\n"))

    def test_script_info_with_bom(self) -> None:
        self.assertTrue(is_ass_subtitle("\ufeff[Script Info]\nTitle: x\n"))

    def test_subtitle_file_suffix(self) -> None:
        self.assertTrue(is_subtitle_file(Path("x.ass")))

    def test_dialogue(self) -> None:
        self.assertTrue(is_ass_subtitle("Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Hi"))

    def test_plain_text(self) -> None:
        self.assertFalse(is_ass_subtitle("Hello world"))


class TestPairing(unittest.TestCase):
    def test_zip_links(self) -> None:
        pairs = pair_videos_and_text_links(
            [Path("a.mp4"), Path("b.mp4")],
            ["s1.ass", "s2.ass"],
        )
        self.assertEqual(pairs[0][1], "s1.ass")

    def test_zip(self) -> None:
        pairs = pair_videos_and_texts(
            [Path("a.mp4"), Path("b.mp4")],
            ["one", "two"],
        )
        self.assertEqual(pairs, [(Path("a.mp4"), "one"), (Path("b.mp4"), "two")])

    def test_one_text(self) -> None:
        pairs = pair_videos_and_texts(
            [Path("a.mp4"), Path("b.mp4")],
            ["shared"],
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], "shared")
        self.assertEqual(pairs[1][1], "shared")

    def test_pad_texts(self) -> None:
        pairs = pair_videos_and_texts(
            [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")],
            ["first", "second"],
        )
        self.assertEqual(pairs[2][1], "second")

    def test_empty_texts(self) -> None:
        pairs = pair_videos_and_texts([Path("a.mp4")], [])
        self.assertEqual(pairs, [(Path("a.mp4"), "")])


class TestEmulate(unittest.TestCase):
    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_ADD_SOFT_SUBTITLES"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("subs"))
            bundle = ArrayBundle()
            bundle.videos.append(ctx.new_link("videos", ".mp4", b"v"))
            bundle.texts.append(ctx.new_link("texts", ".txt", "Hello\nWorld"))
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = add_soft_subtitles_run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.texts), 0)
            self.assertEqual(len(out.sounds), 0)
        finally:
            os.environ.pop("AH_EMULATE_ADD_SOFT_SUBTITLES", None)

    def test_no_videos_raises(self) -> None:
        os.environ["AH_EMULATE_ADD_SOFT_SUBTITLES"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("subs"))
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            with self.assertRaises(RuntimeError):
                add_soft_subtitles_run(ctx, inp)
        finally:
            os.environ.pop("AH_EMULATE_ADD_SOFT_SUBTITLES", None)


if __name__ == "__main__":
    unittest.main()
