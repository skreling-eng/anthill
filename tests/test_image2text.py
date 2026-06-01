"""Tests for $image2text external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.image2text.model_list import get_image2text_model
from externals.image2text.model_paths import model_dir, model_ready
from externals.image2text.run import _prompts_for_images, run
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


class TestImage2TextParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$image2text(max_tokens=128)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "image2text")
        self.assertEqual(expr.args.get("max_tokens"), "128")

    def test_parse_model_qwen3(self) -> None:
        expr = parse_actions("$image2text(model='qwen3', gpu=True)")
        self.assertEqual(expr.args.get("model"), "qwen3")
        self.assertEqual(expr.args.get("gpu"), "True")


class TestImage2TextPrompts(unittest.TestCase):
    def test_prompt_arg_repeats(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(session=session, op_dir=session.next_op_dir("i2t"))
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"prompt": "What is this?"},
            prompt_text="",
        )
        self.assertEqual(_prompts_for_images(ctx, inp, 3), ["What is this?"] * 3)

    def test_default_prompt(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(session=session, op_dir=session.next_op_dir("i2t"))
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        prompts = _prompts_for_images(ctx, inp, 2)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Describe", prompts[0])


class TestImage2TextRun(unittest.TestCase):
    def test_emulate_one_text_per_image(self) -> None:
        os.environ["AH_EMULATE_IMAGE2TEXT"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("image2text")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            bundle.prompts.append(
                ctx.new_link("prompts", ".txt", "Name this object.\n")
            )
            bundle.texts.append(ctx.new_link("texts", ".txt", "old text\n"))
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            self.assertEqual(len(out.texts), 1)
            self.assertEqual(out.prompts, [])
            text = ctx.read_link_text(out.texts[0])
            self.assertIn("[emulated $image2text", text)
            self.assertIn("Name this object", text)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2TEXT", None)

    def test_resolve_use_gpu_arg_overrides(self) -> None:
        from externals.image2text.run import _resolve_use_gpu

        inp = ExternalInput(bundle=ArrayBundle(), args={"gpu": "False"}, prompt_text="")
        self.assertFalse(_resolve_use_gpu(inp))

    def test_resolve_use_gpu_env_overrides(self) -> None:
        from externals.image2text.run import _resolve_use_gpu

        os.environ["AH_IMAGE2TEXT_GPU"] = "0"
        try:
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            self.assertFalse(_resolve_use_gpu(inp))
        finally:
            os.environ.pop("AH_IMAGE2TEXT_GPU", None)
        os.environ["AH_EMULATE_IMAGE2TEXT"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("image2text")
            )
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            self.assertIn("no images", ctx.read_link_text(out.texts[0]))
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2TEXT", None)


class TestImage2TextModelPaths(unittest.TestCase):
    def test_model_dir_under_qwen_vl(self) -> None:
        path = model_dir(get_image2text_model("qwen2"))
        self.assertIn("qwen-vl", str(path).replace("\\", "/"))
        self.assertIn("Qwen2-VL-2B-Instruct", path.name)

    def test_model_ready_false_when_missing(self) -> None:
        if model_ready():
            self.skipTest("model already downloaded")
        self.assertFalse(model_ready())


if __name__ == "__main__":
    unittest.main()
