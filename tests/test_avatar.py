"""Tests for $avatar external."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from externals.avatar.comfy_workflow import build_avatar_prompt, resolve_avatar_size
from externals.avatar.model_paths import default_attention_mode, resolve_blocks_to_swap
from externals.avatar.model_paths import (
    configure_avatar_tiled_vae_for_job,
    fit_avatar_resolution,
    resolve_defer_transformer_load,
    resolve_drop_frames,
    resolve_motion_frame,
    resolve_seam_blend_frames,
    resolve_tiled_vae,
)
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
                self.assertEqual(wan["inputs"]["load_device"], "offload_device")
                swap = next(
                    n for n in wf.values() if n.get("class_type") == "WanVideoBlockSwap"
                )
                swap_val = resolve_blocks_to_swap()
                self.assertEqual(swap["inputs"]["blocks_to_swap"], swap_val)
                if swap_val == 0:
                    self.assertNotIn("block_swap_args", wan["inputs"])
                else:
                    self.assertIn("block_swap_args", wan["inputs"])
        finally:
            os.environ.pop("AVATAR_ATTENTION_MODE", None)

    def test_blocks_to_swap_override(self) -> None:
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
                blocks_to_swap=15,
            )
            swap = next(
                n for n in wf.values() if n.get("class_type") == "WanVideoBlockSwap"
            )
            self.assertEqual(swap["inputs"]["blocks_to_swap"], 15)
            sampler = next(
                n for n in wf.values() if n.get("class_type") == "WanVideoSamplerv2"
            )
            self.assertTrue(sampler["inputs"]["force_offload"])

    def test_build_avatar_prompt_patches_skyreels_window_defaults(self) -> None:
        os.environ.pop("AVATAR_MOTION_FRAME", None)
        os.environ.pop("AVATAR_DROP_FRAMES", None)
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
            skyreels = next(
                n
                for n in wf.values()
                if n.get("class_type") == "WanVideoImageToVideoSkyreelsv3_audio"
            )
            self.assertEqual(skyreels["inputs"]["motion_frame"], 13)
            self.assertEqual(skyreels["inputs"]["drop_frames"], 2)
            self.assertEqual(skyreels["inputs"]["seam_blend_frames"], 8)

    def test_resolve_motion_frame_and_drop_frames(self) -> None:
        os.environ.pop("AVATAR_MOTION_FRAME", None)
        os.environ.pop("AVATAR_DROP_FRAMES", None)
        os.environ.pop("AVATAR_SEAM_BLEND", None)
        self.assertEqual(resolve_motion_frame(), 13)
        self.assertEqual(resolve_drop_frames(), 2)
        self.assertEqual(resolve_seam_blend_frames(), 8)
        self.assertEqual(resolve_motion_frame(12), 12)
        self.assertEqual(resolve_drop_frames(0), 0)
        os.environ["AVATAR_MOTION_FRAME"] = "7"
        os.environ["AVATAR_DROP_FRAMES"] = "2"
        try:
            self.assertEqual(resolve_motion_frame(), 7)
            self.assertEqual(resolve_drop_frames(), 2)
        finally:
            os.environ.pop("AVATAR_MOTION_FRAME", None)
            os.environ.pop("AVATAR_DROP_FRAMES", None)

    def test_build_avatar_prompt_wires_reference_video(self) -> None:
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
                use_reference_video=True,
            )
            skyreels = next(
                n
                for n in wf.values()
                if n.get("class_type") == "WanVideoImageToVideoSkyreelsv3_audio"
            )
            self.assertEqual(skyreels["inputs"]["reference_video"], ["avatar_reference", 0])
            self.assertEqual(wf["avatar_reference"]["class_type"], "AnthillAvatarReferenceVideo")

    def test_tiled_vae_defaults_false_in_workflow(self) -> None:
        os.environ.pop("AVATAR_TILED_VAE", None)
        os.environ["WAN_I2V_TILED_VAE"] = "1"
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
                skyreels = next(
                    n
                    for n in wf.values()
                    if n.get("class_type") == "WanVideoImageToVideoSkyreelsv3_audio"
                )
                self.assertFalse(skyreels["inputs"]["tiled_vae"])
        finally:
            os.environ.pop("WAN_I2V_TILED_VAE", None)

    def test_configure_avatar_tiled_vae_clears_wan_inheritance(self) -> None:
        os.environ["WAN_I2V_TILED_VAE"] = "1"
        os.environ.pop("AVATAR_TILED_VAE", None)
        try:
            override = configure_avatar_tiled_vae_for_job({})
            self.assertIsNone(override)
            self.assertFalse(resolve_tiled_vae())
            self.assertEqual(os.environ.get("WAN_I2V_TILED_VAE"), "0")
        finally:
            os.environ.pop("WAN_I2V_TILED_VAE", None)
            os.environ.pop("AVATAR_TILED_VAE", None)

    def test_fit_avatar_resolution_caps_large_portraits(self) -> None:
        os.environ["AVATAR_MAX_AREA"] = str(480 * 832)
        try:
            capped_w, capped_h = fit_avatar_resolution(1024, 1024)
            self.assertLessEqual(capped_w * capped_h, 480 * 832)
            self.assertEqual(capped_w % 16, 0)
            self.assertEqual(capped_h % 16, 0)
        finally:
            os.environ.pop("AVATAR_MAX_AREA", None)

    def test_resolve_avatar_size_uses_input_image_without_cap(self) -> None:
        os.environ.pop("AVATAR_MAX_AREA", None)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "face.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
            with unittest.mock.patch(
                "externals.avatar.comfy_workflow.read_image_size",
                return_value=(800, 600),
            ):
                out_w, out_h = resolve_avatar_size(image, width=None, height=None)
            self.assertEqual((out_w, out_h), (800, 592))

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

    def test_defer_dit_load_on_16gb(self) -> None:
        os.environ.pop("AVATAR_DEFER_DIT_LOAD", None)
        try:
            from unittest.mock import patch

            with patch(
                "externals.avatar.model_paths._gpu_vram_gb", return_value=16.0
            ):
                self.assertTrue(resolve_defer_transformer_load(0, True))
        finally:
            os.environ.pop("AVATAR_DEFER_DIT_LOAD", None)

    def test_defer_dit_load_off_on_24gb(self) -> None:
        os.environ.pop("AVATAR_DEFER_DIT_LOAD", None)
        try:
            from unittest.mock import patch

            with patch(
                "externals.avatar.model_paths._gpu_vram_gb", return_value=24.0
            ):
                self.assertFalse(resolve_defer_transformer_load(0, True))
        finally:
            os.environ.pop("AVATAR_DEFER_DIT_LOAD", None)

    def test_defer_dit_load_env_override(self) -> None:
        os.environ["AVATAR_DEFER_DIT_LOAD"] = "1"
        try:
            from unittest.mock import patch

            with patch(
                "externals.avatar.model_paths._gpu_vram_gb", return_value=16.0
            ):
                self.assertTrue(resolve_defer_transformer_load(0, True))
        finally:
            os.environ.pop("AVATAR_DEFER_DIT_LOAD", None)


if __name__ == "__main__":
    unittest.main()
