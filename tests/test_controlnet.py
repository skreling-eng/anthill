"""Tests for $controlnet bundle logic and emulated run."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from ahlib.label_utils import add_label_for_elements, make_label_entry
from externals.api import ExternalContext, ExternalInput
from externals.controlnet.bundle_logic import (
    control_combos,
    source_image_links,
    validate_bundle,
)
from externals.controlnet.comfy_workflow import build_controlnet_workflow, per_control_strength
from externals.controlnet.run import run as controlnet_run


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    r, g, b = rgb
    raw = b"".join(
        b"\x00" + bytes([r, g, b]) * width for _ in range(height)
    )
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


class TestControlnetBundleLogic(unittest.TestCase):
    def test_zip_control_combos_by_index(self) -> None:
        bundle = ArrayBundle(
            images=["src.png", "pose0.png", "pose1.png", "depth0.png", "canny0.png", "canny1.png"],
            labels=[
                make_label_entry("pose", [("images", "pose0.png")]),
                make_label_entry("pose", [("images", "pose1.png")]),
                make_label_entry("depth", [("images", "depth0.png")]),
                make_label_entry("canny", [("images", "canny0.png")]),
                make_label_entry("canny", [("images", "canny1.png")]),
            ],
        )
        self.assertEqual(source_image_links(bundle), [])
        combos = control_combos(bundle)
        self.assertEqual(len(combos), 2)
        self.assertEqual(combos[0].pose, "pose0.png")
        self.assertEqual(combos[0].depth, "depth0.png")
        self.assertEqual(combos[0].canny, "canny0.png")
        self.assertEqual(combos[1].pose, "pose1.png")
        self.assertIsNone(combos[1].depth)
        self.assertEqual(combos[1].canny, "canny1.png")

    def test_explicit_source_label(self) -> None:
        bundle = ArrayBundle(
            images=["src.png", "pose0.png"],
            labels=[
                make_label_entry("source", [("images", "src.png")]),
                make_label_entry("pose", [("images", "pose0.png")]),
            ],
        )
        self.assertEqual(source_image_links(bundle), ["src.png"])

    def test_pose_and_canny_only(self) -> None:
        bundle = ArrayBundle(
            images=["src.png", "pose0.png", "canny0.png"],
            labels=[
                make_label_entry("pose", [("images", "pose0.png")]),
                make_label_entry("canny", [("images", "canny0.png")]),
            ],
        )
        combos = control_combos(bundle)
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0].items(), [("openpose", "pose0.png"), ("canny", "canny0.png")])

    def test_prompt_only_bundle_no_source_required(self) -> None:
        bundle = ArrayBundle(
            images=["pose0.png", "depth0.png"],
            labels=[
                make_label_entry("pose", [("images", "pose0.png")]),
                make_label_entry("depth", [("images", "depth0.png")]),
            ],
        )
        self.assertEqual(source_image_links(bundle), [])
        sources, combos = validate_bundle(bundle)
        self.assertEqual(sources, [])
        self.assertEqual(len(combos), 1)


class TestControlnetWorkflow(unittest.TestCase):
    def test_per_control_strength_scales_with_count(self) -> None:
        self.assertEqual(per_control_strength(0.75, 1), 0.75)
        self.assertLess(per_control_strength(0.75, 3), 0.5)

    def test_empty_sd3_latent_handler_registered(self) -> None:
        import externals.controlnet.comfy_executor  # noqa: F401 — register handlers
        from externals.comfy_inprocess.executor import _NODE_HANDLERS

        self.assertIn("EmptySD3LatentImage", _NODE_HANDLERS)
        self.assertTrue(callable(_NODE_HANDLERS["EmptySD3LatentImage"]))

    def test_txt2img_workflow_uses_empty_latent_and_prompt_encode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            pose = input_dir / "pose.png"
            pose.write_bytes(_png_bytes(64, 64))
            workflow, _ = build_controlnet_workflow(
                prompt="portrait",
                negative_prompt="",
                source_image_path=None,
                control_images=[("openpose", pose)],
                input_dir=input_dir,
                seed=1,
                width=64,
                height=64,
                steps=20,
                cfg=2.5,
                denoise=1.0,
                strength=0.9,
            )
            class_types = {
                node.get("class_type")
                for node in workflow.values()
                if isinstance(node, dict)
            }
            self.assertIn("EmptySD3LatentImage", class_types)
            self.assertNotIn("AnthillTextEncodeQwenImageSource", class_types)
            self.assertNotIn("VAEEncode", class_types)

    def test_reference_workflow_uses_empty_latent_with_source_encode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            src = input_dir / "src.png"
            pose = input_dir / "pose.png"
            src.write_bytes(_png_bytes(64, 64))
            pose.write_bytes(_png_bytes(64, 64))
            workflow, _ = build_controlnet_workflow(
                prompt="portrait",
                negative_prompt="",
                source_image_path=src,
                control_images=[("openpose", pose)],
                input_dir=input_dir,
                seed=1,
                width=64,
                height=64,
                steps=20,
                cfg=2.5,
                denoise=1.0,
                strength=0.9,
            )
            class_types = {
                node.get("class_type")
                for node in workflow.values()
                if isinstance(node, dict)
            }
            self.assertIn("EmptySD3LatentImage", class_types)
            self.assertIn("AnthillTextEncodeQwenImageSource", class_types)
            self.assertNotIn("VAEEncode", class_types)

    def test_img2img_workflow_uses_source_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            src = input_dir / "src.png"
            pose = input_dir / "pose.png"
            src.write_bytes(_png_bytes(64, 64))
            pose.write_bytes(_png_bytes(64, 64))
            workflow, _ = build_controlnet_workflow(
                prompt="portrait",
                negative_prompt="",
                source_image_path=src,
                control_images=[("openpose", pose)],
                input_dir=input_dir,
                seed=1,
                width=64,
                height=64,
                steps=20,
                cfg=2.5,
                denoise=0.55,
                strength=0.75,
            )
            class_types = {
                node.get("class_type")
                for node in workflow.values()
                if isinstance(node, dict)
            }
            self.assertIn("ModelSamplingAuraFlow", class_types)
            self.assertIn("AnthillTextEncodeQwenImageSource", class_types)
            self.assertIn("VAEEncode", class_types)
            pos = next(
                node
                for node in workflow.values()
                if isinstance(node, dict)
                and node.get("class_type") == "AnthillTextEncodeQwenImageSource"
            )
            self.assertIn("image", pos["inputs"])
            self.assertEqual(pos["inputs"]["image"][1], 0)
            sampler = next(
                node
                for node in workflow.values()
                if isinstance(node, dict) and node.get("class_type") == "KSampler"
            )
            self.assertEqual(sampler["inputs"]["scheduler"], "simple")

    def test_three_controls_use_reduced_strength(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            paths = []
            for name in ("pose", "depth", "canny"):
                path = input_dir / f"{name}.png"
                path.write_bytes(_png_bytes(64, 64))
                paths.append(path)
            workflow, _ = build_controlnet_workflow(
                prompt="portrait",
                negative_prompt="",
                source_image_path=None,
                control_images=[
                    ("openpose", paths[0]),
                    ("depth", paths[1]),
                    ("canny", paths[2]),
                ],
                input_dir=input_dir,
                seed=1,
                width=64,
                height=64,
                steps=20,
                cfg=2.5,
                denoise=1.0,
                strength=0.75,
            )
            apply_nodes = [
                node
                for node in workflow.values()
                if isinstance(node, dict) and node.get("class_type") == "ControlNetApplyAdvanced"
            ]
            self.assertEqual(len(apply_nodes), 3)
            self.assertLess(apply_nodes[0]["inputs"]["strength"], 0.5)

    def test_first_controlnet_apply_uses_negative_slot_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            src = input_dir / "src.png"
            pose = input_dir / "pose.png"
            src.write_bytes(_png_bytes(64, 64))
            pose.write_bytes(_png_bytes(64, 64))
            workflow, _ = build_controlnet_workflow(
                prompt="portrait",
                negative_prompt="",
                source_image_path=src,
                control_images=[("openpose", pose)],
                input_dir=input_dir,
                seed=1,
                width=64,
                height=64,
                steps=4,
                cfg=4.0,
                denoise=0.75,
                strength=0.9,
            )
            apply_nodes = [
                node
                for node in workflow.values()
                if isinstance(node, dict) and node.get("class_type") == "ControlNetApplyAdvanced"
            ]
            self.assertEqual(len(apply_nodes), 1)
            neg_link = apply_nodes[0]["inputs"]["negative"]
            self.assertEqual(neg_link[1], 0)


class TestControlnetExternal(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_CONTROLNET"] = "1"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def tearDown(self) -> None:
        for key in ("AH_EMULATE_CONTROLNET", "AH_EXTERNAL_SUBPROCESS"):
            os.environ.pop(key, None)

    def test_emulate_prompt_only_generates_one_per_combo(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("controlnet")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        pose0 = session.new_link(op_dir, "images", ".png", _png_bytes(32, 32))
        depth0 = session.new_link(op_dir, "images", ".png", _png_bytes(32, 32))
        bundle = ArrayBundle(images=[pose0, depth0])
        bundle.labels = [
            make_label_entry("pose", [("images", pose0)]),
            make_label_entry("depth", [("images", depth0)]),
        ]

        out = controlnet_run(
            ctx,
            ExternalInput(
                bundle=bundle,
                args={},
                prompt_text="studio portrait",
            ),
        )
        self.assertEqual(len(out.images), 1)
        self.assertFalse(out.prompts)

    def test_emulate_generates_source_times_combos(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("controlnet")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        src = session.new_link(op_dir, "images", ".png", _png_bytes(64, 64))
        pose0 = session.new_link(op_dir, "images", ".png", _png_bytes(32, 32))
        depth0 = session.new_link(op_dir, "images", ".png", _png_bytes(32, 32))
        bundle = ArrayBundle(images=[src, pose0, depth0])
        bundle = add_label_for_elements(
            add_label_for_elements(bundle, "pose"), "depth"
        )
        # add_label tags all images — fix labels manually for test
        bundle.labels = [
            make_label_entry("source", [("images", src)]),
            make_label_entry("pose", [("images", pose0)]),
            make_label_entry("depth", [("images", depth0)]),
        ]

        out = controlnet_run(
            ctx,
            ExternalInput(
                bundle=bundle,
                args={},
                prompt_text="studio portrait",
            ),
        )
        self.assertEqual(len(out.images), 1)
        self.assertFalse(out.prompts)


if __name__ == "__main__":
    unittest.main()
