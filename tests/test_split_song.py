"""Tests for $split_song split logic and emulate mode."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.split_song.run import run
from externals.split_song.split_logic import find_split_segments, rms_envelope
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestSplitSongLogic(unittest.TestCase):
    def test_segments_respect_period_bounds(self) -> None:
        sr = 100
        duration = 100.0
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        mono = (0.2 + 0.8 * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)
        activity, frame_sec = rms_envelope(mono, sr, frame_sec=1.0)
        segments = find_split_segments(duration, activity, frame_sec, period_sec=20.0)
        self.assertTrue(segments)
        self.assertEqual(segments[0][0], 0.0)
        self.assertEqual(segments[-1][1], duration)
        min_len = 10.0
        max_len = 20.0 + 1e-6
        for start, end in segments:
            length = end - start
            if end < duration - 1e-6:
                self.assertGreaterEqual(length, min_len - 1e-6)
                self.assertLessEqual(length, max_len)
            else:
                self.assertLessEqual(length, max_len + 1e-6)

    def test_short_audio_single_segment(self) -> None:
        activity = np.array([0.1, 0.2, 0.05], dtype=np.float64)
        segments = find_split_segments(8.0, activity, 1.0, period_sec=30.0)
        self.assertEqual(segments, [(0.0, 8.0)])


class TestSplitSongExternal(unittest.TestCase):
    def test_emulate_split_song(self) -> None:
        os.environ["AH_EMULATE_SPLIT_SONG"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("split_song")
            sound_path = op_dir / "sounds" / "song.wav"
            sound_path.parent.mkdir(parents=True, exist_ok=True)
            sound_path.write_bytes(b"RIFF" + b"\x00" * 40)
            rel_sound = str(sound_path.relative_to(session_dir)).replace("\\", "/")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(sounds=[rel_sound]),
                args={"period": "30"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.sounds), 2)
        finally:
            os.environ.pop("AH_EMULATE_SPLIT_SONG", None)


if __name__ == "__main__":
    unittest.main()
