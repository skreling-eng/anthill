"""Tests for $comfy INPUT_* placeholder patching."""

from __future__ import annotations

import unittest

from externals.comfy.client import (
    IMAGE_PLACEHOLDERS,
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_IMAGE_ALT,
    PLACEHOLDER_SEED,
    patch_placeholders,
    patch_seed_placeholder,
    resolve_workflow_path,
    workflow_contains_any,
)


class TestResolveWorkflowPath(unittest.TestCase):
    def test_finds_json_in_comfy_workflows(self) -> None:
        path = resolve_workflow_path("Qwen-Rapid-AIO_4.json")
        self.assertTrue(path.is_file())
        self.assertIn("comfy_workflows", str(path).replace("\\", "/"))


class TestImagePlaceholders(unittest.TestCase):
    def test_workflow_contains_bare_or_with_extension(self) -> None:
        self.assertTrue(
            workflow_contains_any({"image": "INPUT_IMAGE"}, IMAGE_PLACEHOLDERS)
        )
        self.assertTrue(
            workflow_contains_any({"image": "INPUT_IMAGE.png"}, IMAGE_PLACEHOLDERS)
        )
        self.assertFalse(
            workflow_contains_any({"image": "other.png"}, IMAGE_PLACEHOLDERS)
        )

    def test_patch_input_image_bare(self) -> None:
        wf = {"load": {"inputs": {"image": "INPUT_IMAGE"}}}
        out = patch_placeholders(
            wf,
            {PLACEHOLDER_IMAGE: "uploaded.png", PLACEHOLDER_IMAGE_ALT: "uploaded.png"},
        )
        self.assertEqual(out["load"]["inputs"]["image"], "uploaded.png")

    def test_patch_input_image_png(self) -> None:
        wf = {"load": {"inputs": {"image": "INPUT_IMAGE.png"}}}
        out = patch_placeholders(
            wf,
            {PLACEHOLDER_IMAGE: "uploaded.png", PLACEHOLDER_IMAGE_ALT: "uploaded.png"},
        )
        self.assertEqual(out["load"]["inputs"]["image"], "uploaded.png")

    def test_longer_placeholder_replaced_first(self) -> None:
        wf = {"x": "INPUT_IMAGE.png"}
        out = patch_placeholders(
            wf,
            {PLACEHOLDER_IMAGE_ALT: "wrong.png", PLACEHOLDER_IMAGE: "right.png"},
        )
        self.assertEqual(out["x"], "right.png")


class TestSeedPlaceholder(unittest.TestCase):
    def test_patch_seed_legacy_list_form_becomes_scalar(self) -> None:
        wf = {"8": {"inputs": {"seed": [PLACEHOLDER_SEED, 0]}}}
        out = patch_seed_placeholder(wf, 42_000_000)
        self.assertEqual(out["8"]["inputs"]["seed"], 42_000_000)

    def test_patch_seed_string_placeholder(self) -> None:
        wf = {"8": {"inputs": {"seed": PLACEHOLDER_SEED}}}
        out = patch_seed_placeholder(wf, 99)
        self.assertEqual(out["8"]["inputs"]["seed"], 99)

    def test_patch_seed_plain_string(self) -> None:
        wf = {"seed": PLACEHOLDER_SEED}
        out = patch_seed_placeholder(wf, 1)
        self.assertEqual(out["seed"], 1)

    def test_seed_in_range_constants(self) -> None:
        from externals.comfy.client import SEED_MAX, SEED_MIN

        self.assertEqual(SEED_MIN, 1)
        self.assertEqual(SEED_MAX, 200_000_000)


if __name__ == "__main__":
    unittest.main()
