"""Tests for $image2image external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

import torch

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.image2image.aio_loader import count_aio_tensors
from externals.image2image.model_paths import (
    DEFAULT_CKPT,
    DEFAULT_MODEL,
    MODEL_ALIASES,
    model_ready,
    resolve_checkpoint,
)
from externals import external_handles_repeat
from externals.image2image.run import (
    _align_prompts,
    _build_edit_jobs,
    _jobs_share_encode,
    _read_prompt_list,
    run,
)
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (255, 255, 255)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 "
            b"\x08\x02\x00\x00\x00\xfc\x18\xed\xa3\x00\x00\x00\x0cIDATx\x9cc``\x00"
            b"\x00\x00\x04\x00\x01\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class TestImage2ImageRepeat(unittest.TestCase):
    def test_handles_repeat_natively(self) -> None:
        self.assertTrue(external_handles_repeat("image2image"))

    def test_align_prompts(self) -> None:
        self.assertEqual(_align_prompts(["a"], 3), ["a", "a", "a"])
        self.assertEqual(_align_prompts(["a", "b"], 3), ["a", "b", "b"])

    def test_jobs_share_encode(self) -> None:
        paths = [Path("a.png")]
        jobs = _build_edit_jobs(
            image_paths=paths,
            prompts=["edit"],
            use_all=False,
            repeat=3,
        )
        self.assertTrue(_jobs_share_encode(jobs))

    def test_build_jobs_with_repeat(self) -> None:
        paths = [Path("a.png"), Path("b.png")]
        jobs = _build_edit_jobs(
            image_paths=paths,
            prompts=["edit"],
            use_all=False,
            repeat=2,
        )
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0], ([paths[0]], "edit"))
        self.assertEqual(jobs[1], ([paths[0]], "edit"))

    def test_read_prompt_list_once_from_bundle(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(session=session, op_dir=session.next_op_dir("image2image"))
        bundle = ArrayBundle()
        bundle.prompts.append(ctx.new_link("prompts", ".txt", "anime style\n"))
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="ignored if prompts[]")
        self.assertEqual(_read_prompt_list(ctx, inp), ["anime style"])

    def test_emulate_repeat(self) -> None:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("image2image")
            )
            bundle = ArrayBundle()
            bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            inp = ExternalInput(
                bundle=bundle,
                args={},
                prompt_text="",
                repeat=3,
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 3)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)


class TestImage2ImageParse(unittest.TestCase):
    def test_parse_use_all(self) -> None:
        expr = parse_actions("$image2image(use_all=True, steps=4)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "image2image")
        self.assertEqual(expr.args.get("use_all"), "True")
        self.assertEqual(expr.args.get("steps"), "4")

    def test_parse_repeat(self) -> None:
        expr = parse_actions("$image2image[3]")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "image2image")
        self.assertEqual(expr.repeat, 3)


class TestImage2ImageModels(unittest.TestCase):
    def test_model_aliases(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "sfw-v23")
        self.assertEqual(DEFAULT_CKPT, MODEL_ALIASES["sfw-v23"])
        self.assertEqual(
            MODEL_ALIASES["nsfw-v23"],
            "Qwen-Rapid-AIO-NSFW-v23.safetensors",
        )
        if model_ready("sfw-v23"):
            path = resolve_checkpoint("sfw-v23")
            self.assertEqual(path.name, "Qwen-Rapid-AIO-SFW-v23.safetensors")
        if model_ready("nsfw-v23"):
            path = resolve_checkpoint("nsfw-v23")
            self.assertEqual(path.name, "Qwen-Rapid-AIO-NSFW-v23.safetensors")


class TestAioLoader(unittest.TestCase):
    def test_count_aio_tensors(self) -> None:
        if not model_ready():
            self.skipTest("checkpoint missing")
        tr, te, vae = count_aio_tensors(resolve_checkpoint(DEFAULT_CKPT))
        self.assertGreater(tr, 1000)
        self.assertGreater(te, 100)
        self.assertGreater(vae, 50)

    def test_aio_weights_map_to_modules(self) -> None:
        if not model_ready():
            self.skipTest("checkpoint missing")
        try:
            from accelerate import init_empty_weights
            from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
            from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            self.skipTest("torch/diffusers not installed")

        from externals.image2image.aio_loader import apply_aio_checkpoint, finalize_module
        from externals.image2image.qwen_pipeline import base_model_dir, ensure_base_assets

        ensure_base_assets()
        base = base_model_dir()
        ckpt = resolve_checkpoint(DEFAULT_CKPT)
        dtype = torch.bfloat16

        with init_empty_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                base / "transformer" / "config.json"
            )
            text_encoder = Qwen2_5_VLForConditionalGeneration(
                AutoConfig.from_pretrained(base / "text_encoder")
            )
            vae = AutoencoderKLQwenImage.from_config(base / "vae" / "config.json")

        apply_aio_checkpoint(
            aio_path=ckpt,
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
        )
        finalize_module(transformer, dtype=dtype, label="transformer")
        finalize_module(text_encoder, dtype=dtype, label="text_encoder")
        finalize_module(vae, dtype=dtype, label="vae")

class TestImage2ImageRun(unittest.TestCase):
    def test_emulate_iterate(self) -> None:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("image2image")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            for _ in range(2):
                bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            bundle.prompts.append(ctx.new_link("prompts", ".txt", "fix hands\n"))
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 2)
            self.assertEqual(out.prompts, [])
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)

    def test_emulate_use_all(self) -> None:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("image2image")
            )
            bundle = ArrayBundle()
            for _ in range(3):
                bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            inp = ExternalInput(
                bundle=bundle,
                args={"use_all": "True"},
                prompt_text="combine into one portrait",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            text = ctx.read_link_text(out.images[0])
            self.assertIn("use_all=True", text)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)


class TestImage2ImageModelPaths(unittest.TestCase):
    def test_default_checkpoint_exists(self) -> None:
        if not model_ready():
            self.skipTest("Qwen-Rapid checkpoint not installed")
        path = resolve_checkpoint(DEFAULT_CKPT)
        self.assertTrue(path.is_file())
        self.assertIn("qwen-rapid", str(path).replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
