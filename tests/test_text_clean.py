"""Tests for LLM display text cleanup."""

from __future__ import annotations

import unittest

from ahlib.text_clean import clean_display_text


class TestCleanDisplayText(unittest.TestCase):
    def test_strips_end_of_turn(self) -> None:
        self.assertEqual(
            clean_display_text("Hello there!<end_of_turn>"),
            "Hello there!",
        )

    def test_strips_qwen_tokens(self) -> None:
        raw = "Hi<|im_end|>\n<|im_start|>assistant\nDone"
        self.assertEqual(clean_display_text(raw), "Hi\nDone")

    def test_strips_start_of_turn_role(self) -> None:
        self.assertEqual(
            clean_display_text("<start_of_turn>model\nHello"),
            "Hello",
        )


if __name__ == "__main__":
    unittest.main()
