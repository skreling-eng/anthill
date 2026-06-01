"""Tests for comfy_lib $image2image backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from externals.image2image.comfy_executor import _topo_order
from externals.image2image.comfy_workflow import (
    build_edit_prompt,
    load_qwen_workflow,
    resolve_output_size,
)
from externals.comfy.client import PLACEHOLDER_PROMPT


class TestOutputSize(unittest.TestCase):
    def test_resolve_from_image_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            img = Path(tmp) / "wide.png"
            Image.new("RGB", (1000, 500), (0, 0, 0)).save(img)
            w, h = resolve_output_size(img, width=None, height=None)
            self.assertEqual((w, h), (1000, 496))

    def test_explicit_overrides_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            img = Path(tmp) / "src.png"
            Image.new("RGB", (640, 480), (0, 0, 0)).save(img)
            w, h = resolve_output_size(img, width=720, height=1280)
            self.assertEqual((w, h), (720, 1280))


class TestComfyWorkflow(unittest.TestCase):
    def test_load_qwen_workflow(self) -> None:
        wf = load_qwen_workflow()
        types = {n.get("class_type") for n in wf.values()}
        self.assertIn("CheckpointLoaderSimple", types)
        self.assertIn("TextEncodeQwenImageEditPlus", types)
        self.assertIn("KSampler", types)

    def test_topo_order(self) -> None:
        wf = load_qwen_workflow()
        order = _topo_order(wf)
        self.assertIn("1", order)
        self.assertIn("2", order)
        self.assertTrue(order.index("1") < order.index("2"))

    def test_build_edit_prompt_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "src.png"
            img.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
                b"\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            inp = Path(tmp) / "input"
            wf, seed = build_edit_prompt(
                prompt="make it blue",
                image_paths=[img],
                input_dir=inp,
                checkpoint_name="Qwen-Rapid-AIO-SFW-v23.safetensors",
                seed=42,
                width=512,
                height=768,
                steps=4,
            )
            self.assertEqual(seed, 42)
            ks = next(n for n in wf.values() if n.get("class_type") == "KSampler")
            self.assertEqual(ks["inputs"]["steps"], 4)
            self.assertEqual(ks["inputs"]["seed"], 42)
            pos = next(
                n
                for n in wf.values()
                if n.get("class_type") == "TextEncodeQwenImageEditPlus"
                and n["inputs"].get("prompt") == "make it blue"
            )
            self.assertIn("image1", pos["inputs"])
            self.assertNotIn(PLACEHOLDER_PROMPT, str(wf))


class TestLegacyExecutorKwargs(unittest.TestCase):
    def test_execute_prompt_legacy_forwards_partial_run(self) -> None:
        import inspect

        from externals.comfy_inprocess.executor import execute_prompt_legacy

        params = inspect.signature(execute_prompt_legacy).parameters
        self.assertIn("stop_before_class", params)
        self.assertIn("only_classes", params)
        self.assertIn("initial_outputs", params)


class TestQwenWorkflowDetect(unittest.TestCase):
    def test_prompt_uses_qwen_image_edit(self) -> None:
        from externals.comfy_inprocess.comfy_memory import prompt_uses_qwen_image_edit

        wf = load_qwen_workflow()
        self.assertTrue(prompt_uses_qwen_image_edit(wf))


if __name__ == "__main__":
    unittest.main()
