"""Tests for $detach_audio and $attach_audio."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.attach_audio.run import run as attach_run
from externals.detach_audio.run import run as detach_run
from externals.video_audio.ffmpeg_io import pair_videos_and_sounds
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestParse(unittest.TestCase):
    def test_detach_parse(self) -> None:
        expr = parse_actions("$detach_audio(format='wav')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "detach_audio")

    def test_attach_parse(self) -> None:
        expr = parse_actions("$attach_audio(shortest=1)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "attach_audio")


class TestPairing(unittest.TestCase):
    def test_zip(self) -> None:
        v = [Path("a.mp4"), Path("b.mp4")]
        s = [Path("1.wav"), Path("2.wav")]
        pairs = pair_videos_and_sounds(v, s)
        self.assertEqual(len(pairs), 2)

    def test_one_video(self) -> None:
        pairs = pair_videos_and_sounds([Path("a.mp4")], [Path("1.wav"), Path("2.wav")])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], pairs[1][0])

    def test_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            pair_videos_and_sounds(
                [Path("a.mp4"), Path("b.mp4")],
                [Path("1.wav"), Path("2.wav"), Path("3.wav")],
            )


class TestDetachEmulate(unittest.TestCase):
    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_DETACH_AUDIO"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("detach"))
            bundle = ArrayBundle()
            bundle.videos.append(ctx.new_link("videos", ".mp4", b"fake"))
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = detach_run(ctx, inp)
            self.assertEqual(len(out.sounds), 1)
            self.assertEqual(len(out.videos), 0)
        finally:
            os.environ.pop("AH_EMULATE_DETACH_AUDIO", None)

    def test_no_videos_raises(self) -> None:
        os.environ["AH_EMULATE_DETACH_AUDIO"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("detach"))
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            with self.assertRaises(RuntimeError):
                detach_run(ctx, inp)
        finally:
            os.environ.pop("AH_EMULATE_DETACH_AUDIO", None)


class TestAttachEmulate(unittest.TestCase):
    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_ATTACH_AUDIO"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("attach"))
            bundle = ArrayBundle()
            bundle.videos.append(ctx.new_link("videos", ".mp4", b"v"))
            bundle.sounds.append(ctx.new_link("sounds", ".wav", b"s"))
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = attach_run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.sounds), 0)
            self.assertEqual(len(out.images), 0)
        finally:
            os.environ.pop("AH_EMULATE_ATTACH_AUDIO", None)


if __name__ == "__main__":
    unittest.main()
