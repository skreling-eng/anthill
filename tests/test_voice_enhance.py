"""Tests for $voice_enhance external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.voice_enhance.model_paths import (
    ENHANCER_RUN_DIR,
    checkpoint_ready,
    enhancer_run_dir,
)
from externals.voice_enhance.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestVoiceEnhanceParse(unittest.TestCase):
    def test_parse_args(self) -> None:
        expr = parse_actions("$voice_enhance(device=cuda, denoise_before=1)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "voice_enhance")
        self.assertEqual(expr.args.get("device"), "cuda")
        self.assertEqual(expr.args.get("denoise_before"), "1")


class TestVoiceEnhanceBootstrap(unittest.TestCase):
    def test_hparams_yaml_loads_with_posix_paths(self) -> None:
        from externals.voice_enhance.model_paths import ENHANCER_RUN_DIR
        from externals.voice_enhance.resemble_bootstrap import bootstrap_resemble_inference

        hp_yaml = ENHANCER_RUN_DIR / "hparams.yaml"
        if not hp_yaml.is_file():
            self.skipTest("enhancer hparams.yaml not present")
        try:
            bootstrap_resemble_inference()
            from resemble_enhance.hparams import HParams

            hp = HParams.load(ENHANCER_RUN_DIR)
        except ImportError:
            self.skipTest("resemble-enhance not installed")
        self.assertEqual(hp.wav_rate, 44_100)


class TestVoiceEnhancePaths(unittest.TestCase):
    def test_enhancer_run_dir(self) -> None:
        self.assertTrue(str(enhancer_run_dir()).endswith("enhancer_stage2"))

    def test_checkpoint_ready_false_by_default(self) -> None:
        if checkpoint_ready(ENHANCER_RUN_DIR):
            self.skipTest("checkpoint already present")
        self.assertFalse(checkpoint_ready(ENHANCER_RUN_DIR))


class TestVoiceEnhanceRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get("AH_EMULATE_VOICE_ENHANCE")
        os.environ["AH_EMULATE_VOICE_ENHANCE"] = "1"

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("AH_EMULATE_VOICE_ENHANCE", None)
        else:
            os.environ["AH_EMULATE_VOICE_ENHANCE"] = self._prev

    def test_emulated_run(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("voice_enhance")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(
            bundle=ArrayBundle(sounds=["sounds/vocals.wav"]),
            args={"device": "cpu"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)

    def test_requires_sounds(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("voice_enhance")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(RuntimeError):
            run(ctx, inp)

    @mock.patch("externals.voice_enhance.run.enhance_file")
    def test_enhance_calls_api(self, enhance_fn: mock.MagicMock) -> None:
        os.environ.pop("AH_EMULATE_VOICE_ENHANCE", None)
        import numpy as np

        enhance_fn.return_value = (np.zeros(1000, dtype=np.float32), 44100)
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("voice_enhance")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        wav_link = ctx.new_link("sounds", ".wav", b"\x00\x00\x00\x00")
        inp = ExternalInput(
            bundle=ArrayBundle(sounds=[wav_link]),
            args={"device": "cpu"},
            prompt_text="",
        )
        with mock.patch("externals.voice_enhance.run._require_resemble"):
            out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)
        enhance_fn.assert_called_once()
        os.environ["AH_EMULATE_VOICE_ENHANCE"] = "1"
