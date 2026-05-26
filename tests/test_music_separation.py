"""Tests for $music_separation external."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.music_separation.model_paths import MODEL_ID, model_ready, ensure_model
from externals.music_separation.models import DEFAULT_MODEL, VARIANTS, resolve_variant
from externals.music_separation.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestMusicSeparationParse(unittest.TestCase):
    def test_parse_args(self) -> None:
        expr = parse_actions("$music_separation(device=CPU, shifts=2)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "music_separation")
        self.assertEqual(expr.args.get("device"), "CPU")
        self.assertEqual(expr.args.get("shifts"), "2")

    def test_parse_roformer_model(self) -> None:
        expr = parse_actions("$music_separation(model=bs_roformer_sw)")
        self.assertEqual(expr.args.get("model"), "bs_roformer_sw")


class TestMusicSeparationModels(unittest.TestCase):
    def test_default_is_bs_roformer_sw(self) -> None:
        variant = resolve_variant("")
        self.assertEqual(variant.id, "bs_roformer_sw")
        self.assertEqual(variant.backend, "roformer")

    def test_htdemucs_v4(self) -> None:
        variant = resolve_variant("htdemucs_v4")
        self.assertEqual(variant.backend, "openvino")
        self.assertEqual(len(variant.stems), 4)

    def test_roformer_2stem(self) -> None:
        variant = resolve_variant("bs_roformer_viperx_1297")
        self.assertEqual(variant.backend, "roformer")
        self.assertEqual(variant.stems, ("vocals", "instrumental"))

    def test_roformer_6stem(self) -> None:
        variant = resolve_variant("bs_roformer_sw")
        self.assertEqual(variant.backend, "roformer")
        self.assertEqual(variant.stems, ("vocals", "drums", "bass", "guitar", "piano", "other"))

    def test_aliases(self) -> None:
        self.assertEqual(resolve_variant("2stem").id, "bs_roformer_viperx_1297")
        self.assertEqual(resolve_variant("6stem").id, "bs_roformer_sw")


class TestHtDemucsPipeline(unittest.TestCase):
    def test_magnitude_layout_matches_demucs(self) -> None:
        import torch
        from externals.music_separation.pipeline import _magnitude

        b, c, fr, t = 1, 2, 2048, 336
        z_np = (
            np.random.randn(b, c, fr, t).astype(np.float32)
            + 1j * np.random.randn(b, c, fr, t).astype(np.float32)
        )
        z_t = torch.from_numpy(z_np)
        expected = (
            torch.view_as_real(z_t).permute(0, 1, 4, 2, 3).reshape(b, c * 2, fr, t).numpy()
        )
        got = _magnitude(z_np)
        self.assertTrue(np.allclose(got, expected, atol=1e-5))


class TestRoformerPaths(unittest.TestCase):
    def test_resolve_output_path_under_out_dir(self) -> None:
        from externals.music_separation.roformer import (
            _resolve_output_path,
            _stem_from_output_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            wav = out_dir / "song_(vocals)_model.wav"
            wav.write_bytes(b"RIFF")
            resolved = _resolve_output_path("song_(vocals)_model.wav", out_dir)
            self.assertEqual(resolved, wav.resolve())
            self.assertEqual(
                _stem_from_output_path(resolved, ("vocals", "instrumental")),
                "vocals",
            )


class TestMusicSeparationRuntime(unittest.TestCase):
    def test_emulated_run_default(self) -> None:
        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("music_separation")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/track.wav"])
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), len(VARIANTS[DEFAULT_MODEL].stems))

    def test_emulated_run_roformer_2stem(self) -> None:
        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("music_separation")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/track.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={"model": "bs_roformer_viperx_1297"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 2)

    def test_emulated_run_roformer_6stem(self) -> None:
        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("music_separation")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/track.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={"model": "bs_roformer_sw"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 6)

    def test_emulated_run_all_models(self) -> None:
        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("music_separation")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/track.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={},
            prompt_text="",
            arg_lists={
                "model": [
                    "bs_roformer_sw",
                    "bs_roformer_viperx_1297",
                    "htdemucs_v4",
                ]
            },
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 6 + 2 + 4)

    def test_integration_multi_model_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        os.environ["AH_EMULATE_FILE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,music_separation,list"
        source = """
@models: $list
bs_roformer_sw, bs_roformer_viperx_1297, htdemucs_v4

@stems: $file('song.wav') -> $music_separation(model=@models)
"""
        result, _ = _run(source, "stems")
        self.assertEqual(len(result.sounds), 12)

    def test_requires_sounds(self) -> None:
        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("music_separation")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(RuntimeError):
            run(ctx, inp)

    def test_integration_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        os.environ["AH_EMULATE_MUSIC_SEPARATION"] = "1"
        os.environ["AH_EMULATE_FILE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,music_separation"
        source = """
@track: $file('song.wav')
@stems: @track -> $music_separation
"""
        result, _ = _run(source, "stems")
        self.assertEqual(len(result.sounds), 6)


class TestMusicSeparationModel(unittest.TestCase):
    def test_model_downloaded_in_repo(self) -> None:
        ensure_model()
        self.assertTrue(model_ready(), f"Intel {MODEL_ID} model should be under models/")

    def test_default_shifts_match_audacity(self) -> None:
        from externals.music_separation.pipeline import DEFAULT_SHIFTS

        self.assertEqual(DEFAULT_SHIFTS, 2)


if __name__ == "__main__":
    unittest.main()
