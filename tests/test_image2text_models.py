"""$image2text model registry."""

from __future__ import annotations

import unittest

from externals.image2text.model_list import get_image2text_model
from externals.image2text.model_paths import model_dir


class TestImage2TextModels(unittest.TestCase):
    def test_default_is_qwen2(self) -> None:
        m = get_image2text_model("default")
        self.assertEqual(m.family, "qwen2")
        self.assertIn("Qwen2-VL-2B", m.dir_name())

    def test_qwen3_profile(self) -> None:
        m = get_image2text_model("qwen3")
        self.assertEqual(m.family, "qwen3")
        self.assertEqual(m.hf_repo, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertIn("Qwen3-VL-8B", m.dir_name())

    def test_model_dirs_differ(self) -> None:
        d2 = model_dir("qwen2")
        d3 = model_dir("qwen3")
        self.assertNotEqual(d2, d3)

    def test_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_image2text_model("unknown-model")


if __name__ == "__main__":
    unittest.main()
