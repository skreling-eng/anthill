"""Tests for RVC training → inference checkpoint conversion."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from externals.change_voice.ckpt_convert import (
    convert_training_checkpoint,
    ensure_inference_pth,
    is_inference_checkpoint,
    is_training_checkpoint,
)
from externals.change_voice.model_paths import RVC_MODELS_DIR, resolve_model


class TestCkptConvertDetect(unittest.TestCase):
    def test_muscleman_is_inference(self) -> None:
        pth = RVC_MODELS_DIR / "MuscleMan" / "muscleman.pth"
        if not pth.is_file():
            self.skipTest("MuscleMan model not installed")
        self.assertTrue(is_inference_checkpoint(pth))
        self.assertFalse(is_training_checkpoint(pth))

    def test_freddie_is_training(self) -> None:
        pth = RVC_MODELS_DIR / "FreddieMercury300k" / "G_300000.pth"
        if not pth.is_file():
            self.skipTest("FreddieMercury300k not installed")
        self.assertTrue(is_training_checkpoint(pth))
        self.assertFalse(is_inference_checkpoint(pth))


class TestCkptConvertFreddie(unittest.TestCase):
    def test_applio_training_rejects_convert(self) -> None:
        pth = RVC_MODELS_DIR / "FreddieMercury300k" / "G_300000.pth"
        if not pth.is_file():
            self.skipTest("FreddieMercury300k not installed")
        out = pth.parent / "_test_freddie_infer.pth"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                convert_training_checkpoint(pth, out, model_dir=pth.parent)
            self.assertIn("non-standard", str(ctx.exception).lower())
        finally:
            out.unlink(missing_ok=True)


class TestCkptConvertResolve(unittest.TestCase):
    def test_ensure_skips_when_inference_exists(self) -> None:
        model_dir = RVC_MODELS_DIR / "MuscleMan"
        if not (model_dir / "muscleman.pth").is_file():
            self.skipTest("MuscleMan not installed")
        got = ensure_inference_pth(model_dir)
        self.assertEqual(got, model_dir / "muscleman.pth")

    def test_resolve_uses_inference_not_g(self) -> None:
        model_dir = RVC_MODELS_DIR / "MuscleMan"
        if not (model_dir / "muscleman.pth").is_file():
            self.skipTest("MuscleMan not installed")
        pth, _, name = resolve_model("MuscleMan")
        self.assertEqual(name, "MuscleMan")
        self.assertFalse(pth.stem.startswith("G_"))


    def test_resolve_rejects_training_only_when_convert_off(self) -> None:
        import tempfile

        import torch

        from externals.change_voice import model_paths as mp

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "TestOnlyG"
            model_dir.mkdir()
            g = model_dir / "G_100.pth"
            torch.save({"model": {"enc_p.encoder.attn_layers.0.conv_k.bias": torch.zeros(192)}}, g)
            old = mp.RVC_MODELS_DIR
            try:
                mp.RVC_MODELS_DIR = Path(tmp)
                os.environ["AH_CHANGE_VOICE_AUTO_CONVERT"] = "0"
                with self.assertRaises(RuntimeError) as ctx:
                    resolve_model("TestOnlyG")
                msg = str(ctx.exception).lower()
                self.assertTrue(
                    "training checkpoint" in msg or "inference" in msg,
                    msg,
                )
            finally:
                mp.RVC_MODELS_DIR = old
                os.environ.pop("AH_CHANGE_VOICE_AUTO_CONVERT", None)


if __name__ == "__main__":
    unittest.main()
