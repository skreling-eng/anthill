"""Tests for llama.cpp GGUF conversion helpers."""

from __future__ import annotations

import unittest

from externals.model_ah.llama_cpp_tools import python_outtype_for_quant


class TestLlamaCppTools(unittest.TestCase):
    def test_python_outtype_k_quant_fallback(self) -> None:
        self.assertEqual(python_outtype_for_quant("Q4_K_M"), "q8_0")

    def test_python_outtype_named(self) -> None:
        self.assertEqual(python_outtype_for_quant("q8_0"), "q8_0")
        self.assertEqual(python_outtype_for_quant("f16"), "f16")


if __name__ == "__main__":
    unittest.main()
