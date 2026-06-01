"""ComfyUI PromptExecutor integration (import smoke tests)."""

from __future__ import annotations

import os
import unittest


class TestPromptExecutor(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("AH_COMFY_EXECUTOR", None)

    def test_should_use_default_auto(self) -> None:
        from externals.comfy_inprocess.prompt_executor import should_use_comfy_executor

        os.environ.pop("AH_COMFY_EXECUTOR", None)
        self.assertFalse(should_use_comfy_executor())
        qwen = {"3": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {}}}
        self.assertFalse(should_use_comfy_executor(qwen))
        mega = {"1": {"class_type": "WanVaceToVideo", "inputs": {}}}
        self.assertTrue(should_use_comfy_executor(mega))

    def test_explicit_comfy_for_all_workflows(self) -> None:
        from externals.comfy_inprocess.prompt_executor import should_use_comfy_executor

        os.environ["AH_COMFY_EXECUTOR"] = "comfy"
        qwen = {"3": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {}}}
        self.assertTrue(should_use_comfy_executor(qwen))

    def test_legacy_opt_out(self) -> None:
        from externals.comfy_inprocess.prompt_executor import should_use_comfy_executor

        os.environ["AH_COMFY_EXECUTOR"] = "legacy"
        self.assertFalse(should_use_comfy_executor())
        mega = {"1": {"class_type": "WanVaceToVideo", "inputs": {}}}
        self.assertFalse(should_use_comfy_executor(mega))

    def test_import_execution_stack(self) -> None:
        from externals.comfy_inprocess.bootstrap import comfy_lib_root
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs
        import sys

        root = comfy_lib_root()
        ensure_comfy_import_stubs()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from execution import CacheType, PromptExecutor  # noqa: F401

        self.assertIsNotNone(PromptExecutor)
        self.assertIsNotNone(CacheType.CLASSIC)


if __name__ == "__main__":
    unittest.main()
