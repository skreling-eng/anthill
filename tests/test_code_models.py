"""Tests for $code model registry (externals/code/model_paths.py)."""

from __future__ import annotations

import unittest

from externals.code.model_list import llms
from externals.code.model_paths import (
    CODE_MODELS,
    default_profile,
    get_code_profile,
    resolve_profile_key,
)


class TestCodeModelRegistry(unittest.TestCase):
    def test_default_profile(self) -> None:
        self.assertEqual(default_profile().key, "14b")

    def test_resolve_aliases(self) -> None:
        self.assertEqual(resolve_profile_key("default"), "14b")
        self.assertEqual(resolve_profile_key("1.5b"), "1.5b")
        self.assertEqual(resolve_profile_key("15b"), "1.5b")
        self.assertEqual(resolve_profile_key("Qwen2.5-Coder-14B-Instruct"), "14b")
        self.assertEqual(resolve_profile_key("1.5b_ah_lora"), "1.5b_ah_lora")

    def test_llms_built_from_registry(self) -> None:
        for profile in CODE_MODELS.values():
            for alias in profile.aliases:
                self.assertIn(alias, llms)
                self.assertEqual(llms[alias].gguf, str(profile.model_gguf))

    def test_lora_profile_gguf_name(self) -> None:
        profile = get_code_profile("1.5b_ah_lora")
        self.assertEqual(profile.gguf_name, "model_lora.gguf")
        self.assertFalse(profile.allow_upstream_download)


if __name__ == "__main__":
    unittest.main()
