"""Tests for comfy_lib $image2video backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from externals.image2video.comfy_workflow import (
    build_i2v_prompt_for_model,
    load_i2v_workflow,
    resolve_output_size,
)
from ahlib.ah_runtime import ArrayBundle


class TestI2VOutputSize(unittest.TestCase):
    def test_resolve_from_image_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            img = Path(tmp) / "wide.png"
            Image.new("RGB", (768, 1280), (0, 0, 0)).save(img)
            w, h = resolve_output_size(img, width=None, height=None)
            self.assertEqual((w, h), (768, 1280))

    def test_explicit_overrides_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            img = Path(tmp) / "src.png"
            Image.new("RGB", (640, 480), (0, 0, 0)).save(img)
            w, h = resolve_output_size(img, width=832, height=480)
            self.assertEqual((w, h), (832, 480))


class TestI2VWorkflow(unittest.TestCase):
    def test_load_i2v_workflow(self) -> None:
        wf = load_i2v_workflow()
        types = {n.get("class_type") for n in wf.values()}
        self.assertIn("CheckpointLoaderSimple", types)
        self.assertIn("WanImageToVideo", types)
        self.assertIn("CLIPVisionEncode", types)
        self.assertIn("VAEDecode", types)

    def test_build_i2v_prompt_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "src.png"
            img.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                b"\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
                b"\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            inp = Path(tmp) / "input"
            try:
                wf, seed = build_i2v_prompt_for_model(
                    prompt="woman running",
                    image_path=img,
                    input_dir=inp,
                    model_arg="wan",
                    seed=42,
                    width=768,
                    height=1280,
                    steps=4,
                    num_frames=81,
                )
            except FileNotFoundError:
                self.skipTest("Wan checkpoint not installed")
            self.assertEqual(seed, 42)
            ckpt = next(
                n for n in wf.values() if n.get("class_type") == "CheckpointLoaderSimple"
            )
            self.assertIn("i2v-rapid-aio-v10", ckpt["inputs"]["ckpt_name"])
            wan = next(n for n in wf.values() if n.get("class_type") == "WanImageToVideo")
            self.assertEqual(wan["inputs"]["length"], 81)
            ks = next(n for n in wf.values() if n.get("class_type") == "KSampler")
            self.assertEqual(ks["inputs"]["steps"], 4)
            self.assertEqual(ks["inputs"]["sampler_name"], "sa_solver")
            pos = next(
                n
                for n in wf.values()
                if n.get("class_type") == "CLIPTextEncode"
                and "negative" not in (n.get("_meta") or {}).get("title", "").lower()
            )
            self.assertEqual(pos["inputs"]["text"], "woman running")


class TestImage2VideoRepeat(unittest.TestCase):
    def test_repeat_emulate_produces_n_videos(self) -> None:
        import os

        from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
        from externals.api import ExternalContext, ExternalInput
        from externals.image2video.run import run

        os.environ["AH_EMULATE_IMAGE2VIDEO"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("image2video")
            image_path = op_dir / "images" / "start.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
            rel = str(image_path.relative_to(session_dir)).replace("\\", "/")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(images=[rel], prompts=["move"]),
                args={"seed": "42"},
                prompt_text="move",
                repeat=10,
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 10)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2VIDEO", None)

    def test_seed_offsets_when_explicit(self) -> None:
        from externals.api import ExternalInput
        from externals.image2video.run import _seed_for_output

        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"seed": "42"},
            prompt_text="",
            repeat=10,
        )
        self.assertEqual(_seed_for_output(inp, 0), 42)
        self.assertEqual(_seed_for_output(inp, 9), 51)

    def test_seed_random_when_omitted(self) -> None:
        from externals.api import ExternalInput
        from externals.image2video.run import _seed_for_output

        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={},
            prompt_text="",
            repeat=10,
        )
        seeds = {_seed_for_output(inp, i) for i in range(20)}
        self.assertGreater(len(seeds), 1)


if __name__ == "__main__":
    unittest.main()
