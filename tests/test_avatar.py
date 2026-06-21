"""Tests for $avatar external."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from externals.avatar.comfy_workflow import build_avatar_prompt
from externals.avatar.model_paths import default_attention_mode
from externals.api import ExternalContext, ExternalInput
from externals.avatar.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestAvatarExternal(unittest.TestCase):
    def test_emulate_avatar_direct(self) -> None:
        os.environ["AH_EMULATE_AVATAR"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("avatar")
            image_path = op_dir / "images" / "face.png"
            sound_path = op_dir / "sounds" / "voice.wav"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            sound_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
            sound_path.write_bytes(b"RIFF" + b"\x00" * 40)

            rel_image = str(image_path.relative_to(session_dir)).replace("\\", "/")
            rel_sound = str(sound_path.relative_to(session_dir)).replace("\\", "/")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(
                    images=[rel_image],
                    sounds=[rel_sound],
                    prompts=["speaking to camera"],
                ),
                args={},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(out.images, [])
            self.assertEqual(out.sounds, [])
            self.assertEqual(out.prompts, [])
        finally:
            os.environ.pop("AH_EMULATE_AVATAR", None)

    def test_attention_mode_defaults_to_sdpa_without_sage(self) -> None:
        os.environ["AVATAR_ATTENTION_MODE"] = "sdpa"
        try:
            self.assertEqual(default_attention_mode(), "sdpa")
        finally:
            os.environ.pop("AVATAR_ATTENTION_MODE", None)

    def test_build_avatar_prompt_patches_attention_mode(self) -> None:
        os.environ["AVATAR_ATTENTION_MODE"] = "sdpa"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                image = tmp_path / "face.png"
                audio = tmp_path / "voice.wav"
                image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
                audio.write_bytes(b"RIFF" + b"\x00" * 40)
                wf, _ = build_avatar_prompt(
                    prompt="talk",
                    negative_prompt="bad",
                    image_path=image,
                    audio_path=audio,
                    input_dir=tmp_path / "input",
                    seed=42,
                    width=480,
                    height=832,
                )
                wan = next(
                    n for n in wf.values() if n.get("class_type") == "WanVideoModelLoader"
                )
                self.assertEqual(wan["inputs"]["attention_mode"], "sdpa")
        finally:
            os.environ.pop("AVATAR_ATTENTION_MODE", None)

    def test_load_audio_handler_reads_wav(self) -> None:
        from externals.comfy_inprocess.audio_nodes import _load_audio_handler, register_comfy_audio_handlers
        from externals.comfy_inprocess.bootstrap import bootstrap_comfy

        register_comfy_audio_handlers()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            bootstrap_comfy(input_dir=input_dir, output_dir=output_dir, load_wan_wrapper=False)
            import folder_paths

            folder_paths.set_input_directory(str(input_dir))
            wav = input_dir / "clip.wav"
            try:
                import struct
                import wave

                sr = 16000
                n = sr
                with wave.open(str(wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    for i in range(n):
                        val = int(0.3 * 32767 * __import__("math").sin(2 * __import__("math").pi * 440 * i / sr))
                        wf.writeframes(struct.pack("<h", val))
            except OSError as exc:
                self.skipTest(f"wave write failed: {exc}")
            out = _load_audio_handler({"audio": wav.name})
            self.assertIn("waveform", out[0])
            self.assertEqual(out[0]["sample_rate"], sr)
            self.assertGreater(out[0]["waveform"].numel(), 0)


if __name__ == "__main__":
    unittest.main()
