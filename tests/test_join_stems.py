"""Tests for $join_stems external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.join_stems.run import _mix_tracks, run
from externals.music_separation.audio_io import write_wav_bytes
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestJoinStemsParse(unittest.TestCase):
    def test_parse_defaults(self) -> None:
        expr = parse_actions("$join_stems")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "join_stems")


class TestJoinStemsMix(unittest.TestCase):
    def test_mix_sums_and_normalizes(self) -> None:
        a = np.ones((2, 1000), dtype=np.float32) * 0.8
        b = np.ones((2, 500), dtype=np.float32) * 0.8
        mixed = _mix_tracks([a, b], normalize=True)
        self.assertEqual(mixed.shape, (2, 1000))
        self.assertAlmostEqual(float(np.max(np.abs(mixed))), 0.99, places=2)

    def test_mix_without_normalize_can_clip(self) -> None:
        a = np.ones((2, 100), dtype=np.float32)
        b = np.ones((2, 100), dtype=np.float32)
        mixed = _mix_tracks([a, b], normalize=False)
        self.assertAlmostEqual(float(mixed[0, 0]), 2.0)


class TestJoinStemsRuntime(unittest.TestCase):
    def test_emulated_run(self) -> None:
        os.environ["AH_EMULATE_JOIN_STEMS"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(
            bundle=ArrayBundle(sounds=["sounds/a.wav", "sounds/b.wav"]),
            args={},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)

    def test_requires_sounds(self) -> None:
        os.environ["AH_EMULATE_JOIN_STEMS"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(RuntimeError):
            run(ctx, inp)

    def test_joins_two_wavs(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("join_stems")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        tone_a = (np.sin(np.linspace(0, 8 * np.pi, 8000)) * 0.25).astype(np.float32)
        tone_b = (np.sin(np.linspace(0, 4 * np.pi, 4000)) * 0.25).astype(np.float32)
        stereo_a = np.stack([tone_a, tone_a])
        stereo_b = np.stack([tone_b, tone_b])

        link_a = session.new_link(op_dir, "sounds", ".wav", write_wav_bytes(stereo_a, 44100))
        link_b = session.new_link(op_dir, "sounds", ".wav", write_wav_bytes(stereo_b, 44100))

        inp = ExternalInput(
            bundle=ArrayBundle(sounds=[link_a, link_b]),
            args={},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)
        out_path = session.base_dir / out.sounds[0]
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
