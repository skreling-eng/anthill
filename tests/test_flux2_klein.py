"""Tests for Flux.2 Klein comfy workflows."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from externals.comfy.client import PLACEHOLDER_PROMPT
from externals.flux2_klein.comfy_workflow import build_edit_prompt, build_txt2img_prompt
from externals.flux2_klein.model_paths import is_klein_model
from externals.image2image.comfy_executor import _topo_order
from externals.comfy_inprocess.executor import find_node_id


class TestKleinModelPaths(unittest.TestCase):
    def test_is_klein_model(self) -> None:
        self.assertTrue(is_klein_model("klein-fp8"))
        self.assertTrue(is_klein_model("flux2Klein9bFp8_fp8.safetensors"))
        self.assertFalse(is_klein_model("sfw-v23"))


class TestKleinWorkflow(unittest.TestCase):
    def test_prompt_uses_flux2_klein(self) -> None:
        from externals.comfy_inprocess.comfy_memory import prompt_uses_flux2_klein

        wf, _ = build_txt2img_prompt(
            prompt="test",
            model_arg="klein-fp8",
            seed=1,
            width=512,
            height=512,
            steps=4,
            cfg=4.0,
        )
        self.assertTrue(prompt_uses_flux2_klein(wf))

    def test_build_txt2img_prompt(self) -> None:
        wf, seed = build_txt2img_prompt(
            prompt="a red fox",
            model_arg="klein-fp8",
            seed=99,
            width=768,
            height=512,
            steps=20,
            cfg=4.0,
        )
        self.assertEqual(seed, 99)
        types = {n.get("class_type") for n in wf.values()}
        self.assertIn("UNETLoader", types)
        self.assertIn("CLIPLoader", types)
        self.assertIn("KSampler", types)
        pos = next(n for n in wf.values() if n.get("class_type") == "CLIPTextEncode" and n["inputs"].get("text") == "a red fox")
        self.assertIsNotNone(pos)
        self.assertNotIn(PLACEHOLDER_PROMPT, str(wf))
        order = _topo_order(wf)
        self.assertTrue(order.index("1") < order.index("7"))

    def test_snap_caps_large_edit_when_auto_cap(self) -> None:
        import os
        from unittest.mock import patch

        from externals.flux2_klein.comfy_workflow import klein_max_area, snap_klein_latent_size

        with patch.dict(os.environ, {"AH_FLUX2_KLEIN_AUTO_CAP": "1"}, clear=False):
            mock_props = type("P", (), {"total_memory": 16 * (1024**3)})()
            with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_properties", return_value=mock_props
            ):
                w, h = snap_klein_latent_size(720, 1280)
                max_area = klein_max_area()
                self.assertIsNotNone(max_area)
                self.assertLessEqual(w * h, max_area)  # type: ignore[operator]
                self.assertLess(w, 720)

    def test_txt2img_respects_requested_size(self) -> None:
        wf, _ = build_txt2img_prompt(
            prompt="test",
            model_arg="klein-fp8",
            seed=1,
            width=720,
            height=1280,
            steps=4,
            cfg=4.0,
        )
        latent = next(
            n for n in wf.values() if n.get("class_type") == "EmptyFlux2LatentImage"
        )
        self.assertEqual(latent["inputs"]["width"], 720)
        self.assertEqual(latent["inputs"]["height"], 1280)

    def test_txt2img_encode_before_sampler(self) -> None:
        wf, _ = build_txt2img_prompt(
            prompt="test",
            model_arg="klein-fp8",
            seed=1,
            width=512,
            height=512,
            steps=4,
            cfg=4.0,
        )
        order = _topo_order(wf)
        ks = find_node_id(wf, "KSampler")
        pos = next(
            nid
            for nid, n in wf.items()
            if n.get("class_type") == "CLIPTextEncode"
            and n["inputs"].get("text") == "test"
        )
        self.assertIsNotNone(ks)
        self.assertLess(order.index(pos), order.index(ks))

    def test_distilled_steps_normalized(self) -> None:
        from externals.flux2_klein.model_paths import normalize_klein_steps_cfg

        steps, cfg = normalize_klein_steps_cfg("klein-fp8", 20, 4.0)
        self.assertEqual(steps, 4)
        self.assertEqual(cfg, 1.0)

    def test_build_edit_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "src.png"
            img.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
                b"\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            wf, seed = build_edit_prompt(
                prompt="make it blue",
                image_path=img,
                input_dir=Path(tmp) / "input",
                model_arg="klein-fp8",
                seed=42,
                width=512,
                height=768,
                steps=20,
                cfg=4.0,
            )
            self.assertEqual(seed, 42)
            self.assertIn("ReferenceLatent", {n.get("class_type") for n in wf.values()})
            ks = next(n for n in wf.values() if n.get("class_type") == "KSampler")
            self.assertEqual(ks["inputs"]["steps"], 20)
            unet = next(n for n in wf.values() if n.get("class_type") == "UNETLoader")
            self.assertEqual(unet["inputs"]["weight_dtype"], "fp8_e4m3fn_fast")
            clip = next(n for n in wf.values() if n.get("class_type") == "CLIPLoader")
            self.assertEqual(clip["inputs"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
