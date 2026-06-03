"""Tests for externals.llm.gguf_llm helpers."""

from __future__ import annotations

import unittest

from externals.llm.gguf_llm import _load_error_hint


class TestLoadErrorHint(unittest.TestCase):
    def test_mtp_path(self) -> None:
        exc = ValueError("Failed to load model from file: x.gguf")
        hint = _load_error_hint(
            r"G:\models\llm_user\q\Qwen3.6-35B-A3B-MTP-GGUF\file.gguf",
            exc,
        )
        self.assertIn("MTP", hint)
        self.assertIn("non-MTP", hint)

    def test_ssm_tensor_message(self) -> None:
        exc = ValueError("missing tensor 'blk.40.ssm_conv1d.weight'")
        hint = _load_error_hint(r"G:\models\llm_user\q\weights.gguf", exc)
        self.assertIn("MTP", hint)

    def test_unrelated_error(self) -> None:
        self.assertEqual(_load_error_hint("a.gguf", RuntimeError("nope")), "")


if __name__ == "__main__":
    unittest.main()
