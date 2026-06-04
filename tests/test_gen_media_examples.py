"""Tests for procedural media example generation."""

from __future__ import annotations

import unittest
from pathlib import Path

from tools.gen_bulk_examples import build_image_clip_case, generate_media_cases


class TestMediaExampleGen(unittest.TestCase):
    def test_image_clip_case_uses_image_clip_external(self) -> None:
        _, req, body = build_image_clip_case(1)
        self.assertIn("image clip", req.lower())
        self.assertIn("$image_clip", body)
        self.assertIn("$music", body)
        self.assertIn("$image(", body)
        self.assertNotIn("@answer: $llm", body)

    def test_generate_media_cases_mixed_kinds(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        cases = generate_media_cases(10, repo_root=repo)
        self.assertEqual(len(cases), 10)
        kinds = {stem.split("_")[0] if stem.startswith("image2") else stem.rsplit("_", 1)[0] for stem, _, _ in cases}
        # stems like image_clip_0001 -> check externals in bodies
        bodies = " ".join(b for _, _, b in cases)
        self.assertIn("$image_clip", bodies)
        self.assertIn("$music", bodies)


if __name__ == "__main__":
    unittest.main()
