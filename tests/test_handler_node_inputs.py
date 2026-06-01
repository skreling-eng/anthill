"""Anthill handler nodes must declare INPUT_TYPES for Comfy PromptExecutor."""

from __future__ import annotations

import unittest

import externals.image2image.comfy_executor  # noqa: F401 — registers handler
from externals.comfy_inprocess.executor import handler_input_types
from externals.comfy_inprocess.node_registry import _make_handler_node
from externals.comfy_inprocess.executor import _NODE_HANDLERS


class TestHandlerNodeInputs(unittest.TestCase):
    def test_qwen_encode_handler_declares_prompt(self) -> None:
        spec = handler_input_types("TextEncodeQwenImageEditPlus")
        self.assertIn("prompt", spec.get("required", {}))
        self.assertIn("clip", spec.get("required", {}))
        self.assertIn("image1", spec.get("optional", {}))

        handler = _NODE_HANDLERS["TextEncodeQwenImageEditPlus"]
        cls = _make_handler_node(
            "TextEncodeQwenImageEditPlus", handler, input_types=spec
        )
        declared = cls.INPUT_TYPES()
        self.assertIn("prompt", declared["required"])
        self.assertNotEqual(declared, {"required": {}, "optional": {}})


if __name__ == "__main__":
    unittest.main()
