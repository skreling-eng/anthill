"""Qwen image-edit prompt template / template_end trim."""

from __future__ import annotations

import unittest

from externals.image2image.comfy_qwen_prompt import (
    build_image_prompt_prefix,
    format_qwen_edit_llama_text,
    template_end_from_input_ids,
)


class TestQwenEditPromptText(unittest.TestCase):
    def test_llama_text_contains_user_instruction(self) -> None:
        user = "make this image in the simple anime style"
        image_prompt = build_image_prompt_prefix(1)
        text = format_qwen_edit_llama_text(image_prompt, user)
        self.assertIn(user, text)
        self.assertIn("<|im_end|>", text)
        self.assertIn("<|vision_start|>", image_prompt)

    def test_template_end_from_ids_preserves_user_region(self) -> None:
        try:
            from transformers import Qwen2Tokenizer
        except ImportError:
            self.skipTest("transformers not installed")

        from externals.image2image.qwen_pipeline import base_model_dir, ensure_base_assets

        try:
            ensure_base_assets()
            tok_path = base_model_dir() / "tokenizer"
            if not tok_path.is_dir():
                self.skipTest("Qwen base tokenizer not on disk")
            tokenizer = Qwen2Tokenizer.from_pretrained(str(tok_path))
        except (OSError, RuntimeError) as exc:
            self.skipTest(str(exc))

        user = "make this image in the simple anime style"
        image_prompt = build_image_prompt_prefix(1)
        text = format_qwen_edit_llama_text(image_prompt, user)
        ids = tokenizer(text, return_tensors="pt").input_ids
        end = template_end_from_input_ids(ids)
        kept = tokenizer.decode(ids[0, end:].tolist(), skip_special_tokens=False)
        self.assertIn(user, kept)


if __name__ == "__main__":
    unittest.main()
