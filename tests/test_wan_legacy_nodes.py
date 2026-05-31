"""Stock Wan legacy nodes registered for PromptExecutor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


class TestWanLegacyNodes(unittest.TestCase):
    def test_registers_wan_and_primitive(self) -> None:
        from externals.comfy_inprocess.bootstrap import comfy_lib_root
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        root = comfy_lib_root()
        ensure_comfy_import_stubs()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import nodes
        from externals.comfy_inprocess.wan_legacy_nodes import register_wan_legacy_nodes

        register_wan_legacy_nodes()
        self.assertIn("WanImageToVideo", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("WanVaceToVideo", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("PrimitiveInt", nodes.NODE_CLASS_MAPPINGS)
        cls = nodes.NODE_CLASS_MAPPINGS["WanImageToVideo"]
        self.assertTrue(hasattr(cls, "INPUT_TYPES"))
        self.assertEqual(cls.FUNCTION, "execute")


if __name__ == "__main__":
    unittest.main()
