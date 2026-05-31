"""image2video model registry."""

from __future__ import annotations

import unittest

from externals.image2video.model_list import get_video_model


class TestImage2VideoModels(unittest.TestCase):
    def test_mega_nsfw_profile(self) -> None:
        m = get_video_model("mega-nsfw")
        self.assertEqual(m.name, "mega-nsfw")
        self.assertIn("mega-aio-nsfw", m.checkpoint)

    def test_mega_profile(self) -> None:
        m = get_video_model("mega")
        self.assertIn("mega-aio-v12", m.checkpoint)

    def test_default_profile_is_mega(self) -> None:
        m = get_video_model("default")
        self.assertIn("mega-aio-v12", m.checkpoint)
        self.assertEqual(get_video_model("").checkpoint, m.checkpoint)


if __name__ == "__main__":
    unittest.main()
