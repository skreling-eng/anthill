"""ComfyUI-style memory hooks for in-process workflows."""

from __future__ import annotations

import os
import unittest

from externals.comfy_inprocess.comfy_memory import (
    comfy_memory_enabled,
    prompt_uses_mega_vace,
)


class TestComfyMemory(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("AH_COMFY_MEMORY", None)

    def test_detect_mega(self) -> None:
        prompt = {"1": {"class_type": "WanVaceToVideo", "inputs": {}}}
        self.assertTrue(prompt_uses_mega_vace(prompt))

    def test_default_on_for_mega(self) -> None:
        prompt = {"1": {"class_type": "WanVaceToVideo", "inputs": {}}}
        self.assertTrue(comfy_memory_enabled(prompt))

    def test_off_when_disabled(self) -> None:
        os.environ["AH_COMFY_MEMORY"] = "0"
        prompt = {"1": {"class_type": "WanVaceToVideo", "inputs": {}}}
        self.assertFalse(comfy_memory_enabled(prompt))


if __name__ == "__main__":
    unittest.main()
