"""Tests for $change_voice external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.change_voice.model_paths import (
    DEFAULT_MODEL,
    RVC_MODELS_DIR,
    detect_rvc_version,
    resolve_model,
    resolve_rvc_version,
)
from externals.change_voice.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestChangeVoiceParse(unittest.TestCase):
    def test_parse_args(self) -> None:
        expr = parse_actions(
            "$change_voice(model=MuscleMan, f0up_key=2, protect=0.5, device=cuda:0)"
        )
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "change_voice")
        self.assertEqual(expr.args.get("model"), "MuscleMan")
        self.assertEqual(expr.args.get("f0up_key"), "2")


class TestChangeVoiceModelPaths(unittest.TestCase):
    @staticmethod
    def _write_stub_inference(pth: Path) -> None:
        import torch

        torch.save(
            {
                "weight": {"emb_g.weight": torch.zeros(1, 256)},
                "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 1, 256, 48000],
                "info": "test",
                "sr": "48k",
                "f0": 1,
                "version": "v1",
            },
            pth,
        )

    def test_resolve_explicit_paths(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        pth = repo / "test_data_change_voice_tmp" / "voice.pth"
        idx = repo / "test_data_change_voice_tmp" / "voice.index"
        pth.parent.mkdir(exist_ok=True)
        self._write_stub_inference(pth)
        idx.write_bytes(b"idx")
        try:
            got_pth, got_idx, name = resolve_model(
                "",
                model_path=str(pth),
                index_path=str(idx),
            )
            self.assertEqual(got_pth, pth.resolve())
            self.assertEqual(got_idx, idx.resolve())
            self.assertEqual(name, "voice")
        finally:
            pth.unlink(missing_ok=True)
            idx.unlink(missing_ok=True)
            pth.parent.rmdir()

    def test_resolve_by_model_dir(self) -> None:
        model_dir = RVC_MODELS_DIR / "TestVoice"
        model_dir.mkdir(parents=True, exist_ok=True)
        pth = model_dir / "test.pth"
        idx = model_dir / "test.index"
        self._write_stub_inference(pth)
        idx.write_bytes(b"idx")
        try:
            got_pth, got_idx, name = resolve_model("TestVoice")
            self.assertEqual(got_pth, pth.resolve())
            self.assertEqual(got_idx, idx.resolve())
            self.assertEqual(name, "TestVoice")
        finally:
            pth.unlink(missing_ok=True)
            idx.unlink(missing_ok=True)
            model_dir.rmdir()


class TestChangeVoiceVersionDetect(unittest.TestCase):
    def test_detect_v1_from_emb_phone(self) -> None:
        import torch

        repo = Path(__file__).resolve().parents[1]
        pth = repo / "test_data_change_voice_tmp" / "v1_detect.pth"
        pth.parent.mkdir(exist_ok=True)
        torch.save(
            {"weight": {"enc_p.emb_phone.weight": torch.zeros(192, 256)}},
            pth,
        )
        try:
            self.assertEqual(detect_rvc_version(pth), "v1")
        finally:
            pth.unlink(missing_ok=True)
            pth.parent.rmdir()

    def test_detect_v2_from_emb_phone(self) -> None:
        import torch

        repo = Path(__file__).resolve().parents[1]
        pth = repo / "test_data_change_voice_tmp" / "v2_detect.pth"
        pth.parent.mkdir(exist_ok=True)
        torch.save(
            {"weight": {"enc_p.emb_phone.weight": torch.zeros(192, 768)}},
            pth,
        )
        try:
            self.assertEqual(detect_rvc_version(pth), "v2")
        finally:
            pth.unlink(missing_ok=True)
            pth.parent.rmdir()

    def test_detect_falls_back_to_version_key(self) -> None:
        import torch

        repo = Path(__file__).resolve().parents[1]
        pth = repo / "test_data_change_voice_tmp" / "meta_v1.pth"
        pth.parent.mkdir(exist_ok=True)
        torch.save({"version": "v1"}, pth)
        try:
            self.assertEqual(detect_rvc_version(pth), "v1")
        finally:
            pth.unlink(missing_ok=True)
            pth.parent.rmdir()

    def test_detect_from_config_ssl_dim(self) -> None:
        import json
        import torch

        repo = Path(__file__).resolve().parents[1]
        model_dir = repo / "test_data_change_voice_tmp" / "CfgVoice"
        model_dir.mkdir(parents=True, exist_ok=True)
        pth = model_dir / "G_100.pth"
        cfg = model_dir / "config.json"
        torch.save({"model": {}}, pth)
        cfg.write_text(
            json.dumps({"model": {"ssl_dim": 256}}),
            encoding="utf-8",
        )
        try:
            self.assertEqual(detect_rvc_version(pth), "v1")
        finally:
            pth.unlink(missing_ok=True)
            cfg.unlink(missing_ok=True)
            model_dir.rmdir()

    def test_resolve_explicit_overrides_detect(self) -> None:
        import torch

        repo = Path(__file__).resolve().parents[1]
        pth = repo / "test_data_change_voice_tmp" / "override.pth"
        pth.parent.mkdir(exist_ok=True)
        torch.save(
            {"weight": {"enc_p.emb_phone.weight": torch.zeros(192, 256)}},
            pth,
        )
        try:
            self.assertEqual(resolve_rvc_version(pth, "v2"), "v2")
            self.assertEqual(resolve_rvc_version(pth, ""), "v1")
        finally:
            pth.unlink(missing_ok=True)
            pth.parent.rmdir()


class TestChangeVoiceRuntime(unittest.TestCase):
    def test_emulated_run(self) -> None:
        os.environ["AH_EMULATE_CHANGE_VOICE"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("change_voice")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/vocals.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={"model": DEFAULT_MODEL, "f0up_key": "2"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)

    def test_requires_sounds(self) -> None:
        os.environ["AH_EMULATE_CHANGE_VOICE"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("change_voice")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(RuntimeError):
            run(ctx, inp)

    def test_emulated_multi_model(self) -> None:
        os.environ["AH_EMULATE_CHANGE_VOICE"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("change_voice")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/vocals.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={},
            prompt_text="",
            arg_lists={"model": ["VoiceA", "VoiceB", "VoiceC"]},
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 3)

    def test_emulated_multi_model_multi_input(self) -> None:
        os.environ["AH_EMULATE_CHANGE_VOICE"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("change_voice")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/a.wav", "sounds/b.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={},
            prompt_text="",
            arg_lists={"model": ["VoiceA", "VoiceB"]},
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 4)

    def test_integration_multi_model_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        source = """
@voices: $list
lumine-jp-v2, MuscleMan, BartSimpson

@run: $file('vocals.wav') -> $change_voice(model=@voices)
"""
        os.environ["AH_EMULATE_CHANGE_VOICE"] = "1"
        os.environ["AH_EMULATE_FILE"] = "1"
        result, _ = _run(source, "run", inprocess="file,list,change_voice")
        self.assertEqual(len(result.sounds), 3)


if __name__ == "__main__":
    unittest.main()
