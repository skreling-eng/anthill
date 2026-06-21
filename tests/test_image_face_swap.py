"""Tests for $image_face_swap bundle logic and workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ahlib.label_utils import make_label_entry
from ahlib.ah_runtime import ArrayBundle
from externals.image2image.comfy_executor import _topo_order
from externals.image_face_swap.bundle_logic import (
    DEFAULT_PROMPT,
    face_image_links,
    face_swap_jobs,
    target_image_links,
)
from externals.image_face_swap.comfy_workflow import build_face_swap_prompt


class TestFaceSwapBundle(unittest.TestCase):
    def _bundle(self) -> ArrayBundle:
        return ArrayBundle(
            images=["target.png", "donor.png"],
            labels=[
                make_label_entry("face", [("images", "donor.png")]),
            ],
        )

    def test_target_and_face_links(self) -> None:
        bundle = self._bundle()
        self.assertEqual(target_image_links(bundle), ["target.png"])
        self.assertEqual(face_image_links(bundle), ["donor.png"])

    def test_jobs(self) -> None:
        jobs = face_swap_jobs(self._bundle())
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].target, "target.png")
        self.assertEqual(jobs[0].face, "donor.png")

    def test_missing_face_raises(self) -> None:
        bundle = ArrayBundle(images=["only.png"], labels=[])
        with self.assertRaises(RuntimeError):
            face_swap_jobs(bundle)

    def test_missing_target_raises(self) -> None:
        bundle = ArrayBundle(
            images=["donor.png"],
            labels=[make_label_entry("face", [("images", "donor.png")])],
        )
        with self.assertRaises(RuntimeError):
            face_swap_jobs(bundle)


class TestFaceSwapWorkflow(unittest.TestCase):
    def test_build_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.png"
            face = Path(tmp) / "face.png"
            for path in (target, face):
                path.write_bytes(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
                    b"\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
                    b"\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
                )
            wf, seed = build_face_swap_prompt(
                prompt=DEFAULT_PROMPT,
                target_path=target,
                face_path=face,
                input_dir=Path(tmp) / "input",
                model_arg="klein-fp8",
                seed=7,
                width=512,
                height=768,
                steps=20,
                cfg=4.0,
            )
            self.assertEqual(seed, 7)
            types = {n.get("class_type") for n in wf.values()}
            self.assertIn("ReferenceLatent", types)
            ref_nodes = [n for n in wf.values() if n.get("class_type") == "ReferenceLatent"]
            self.assertEqual(len(ref_nodes), 2)
            order = _topo_order(wf)
            self.assertTrue(order.index("9") < order.index("10"))
            self.assertTrue(order.index("10") < order.index("13"))


if __name__ == "__main__":
    unittest.main()
