"""Tests for $audio_instruct external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.audio_instruct.model_list import get_audio_instruct_model
from unittest.mock import patch

from externals.audio_instruct.model_paths import ensure_model, model_dir, model_ready
from externals.audio_instruct.qwen_audio import _conversation
from externals.audio_instruct.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestAudioInstructParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$audio_instruct(prompt='What is this?')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "audio_instruct")
        self.assertEqual(expr.args.get("prompt"), "What is this?")


class TestAudioInstructRun(unittest.TestCase):
    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_AUDIO_INSTRUCT"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("audio_instruct")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.sounds.append(ctx.new_link("sounds", ".wav", b"RIFF"))
            inp = ExternalInput(
                bundle=bundle,
                args={"prompt": "What do you hear?"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            text = ctx.read_link_text(out.texts[0])
            self.assertIn("[emulated $audio_instruct", text)
            self.assertIn("What do you hear?", text)
        finally:
            os.environ.pop("AH_EMULATE_AUDIO_INSTRUCT", None)

    def test_no_sounds_message(self) -> None:
        os.environ["AH_EMULATE_AUDIO_INSTRUCT"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("audio_instruct")
            )
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertIn("no sounds", ctx.read_link_text(out.texts[0]).lower())
        finally:
            os.environ.pop("AH_EMULATE_AUDIO_INSTRUCT", None)


class TestQwenAudioProcessorKwargs(unittest.TestCase):
    def test_processor_uses_audio_kwarg_not_audios(self) -> None:
        import tempfile

        import numpy as np

        wav = Path(tempfile.mkdtemp()) / "t.wav"
        try:
            import soundfile as sf

            sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
        except ImportError:
            wav.write_bytes(b"RIFF")
        model_dir = Path("models/qwen-audio/Qwen2-Audio-7B-Instruct-4bit")
        if not (model_dir / "config.json").is_file():
            self.skipTest("Qwen2-Audio model not installed locally")
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(str(model_dir))
        conv = _conversation(
            wav.resolve(), "test", system=""
        )
        text = proc.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        audio = np.zeros(int(proc.feature_extractor.sampling_rate), dtype=np.float32)
        bad = proc(text=text, audios=[audio], return_tensors="pt", padding=True)
        self.assertNotIn("input_features", bad)
        good = proc(
            text=text,
            audio=[audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=proc.feature_extractor.sampling_rate,
        )
        self.assertIn("input_features", good)


class TestQwenAudioConversation(unittest.TestCase):
    def test_conversation_uses_audio_url(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = Path(f.name)
            wav.write_bytes(b"RIFF")
        try:
            conv = _conversation(wav, "describe", system="sys")
            user = conv[-1]["content"]
            audio_part = next(p for p in user if p["type"] == "audio")
            self.assertIn("audio_url", audio_part)
            self.assertTrue(Path(audio_part["audio_url"]).is_file())
        finally:
            wav.unlink(missing_ok=True)


class TestAudioInstructModels(unittest.TestCase):
    def test_default_profile(self) -> None:
        profile = get_audio_instruct_model("default")
        self.assertEqual(
            profile.subdir.as_posix(), "qwen-audio/Qwen2-Audio-7B-Instruct-4bit"
        )
        self.assertEqual(profile.hf_repo, "alicekyting/Qwen2-Audio-7B-Instruct-4bit")

    def test_model_dir(self) -> None:
        path = model_dir("default")
        self.assertIn("Qwen2-Audio-7B-Instruct-4bit", str(path).replace("\\", "/"))

    def test_model_ready_when_present(self) -> None:
        if model_ready("default"):
            self.assertTrue((model_dir("default") / "config.json").is_file())

    def test_ensure_model_upstream_after_anthill_miss(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "models"
            model_path = root / "qwen-audio" / "Qwen2-Audio-7B-Instruct-4bit"
            model_path.mkdir(parents=True)

            def fake_ready() -> bool:
                return (model_path / "config.json").is_file()

            with patch(
                "externals.audio_instruct.model_paths.models_roots",
                lambda: (root,),
            ), patch(
                "externals.audio_instruct.model_paths.ensure_anthill_tree",
                side_effect=FileNotFoundError("anthill miss"),
            ), patch(
                "externals.audio_instruct.model_paths._download_upstream",
            ) as upstream:
                def finish_download(_m, path: Path) -> None:
                    (path / "config.json").write_text("{}", encoding="utf-8")
                    (path / "model.safetensors").write_bytes(b"x")

                upstream.side_effect = finish_download
                result = ensure_model("default")
                upstream.assert_called_once()
                self.assertEqual(result, model_path)


if __name__ == "__main__":
    unittest.main()
