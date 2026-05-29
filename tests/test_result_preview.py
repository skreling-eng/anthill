"""Tests for action log result preview rendering in app.Interface."""

from __future__ import annotations

import unittest
from pathlib import Path

from app import Interface


class TestResultPreview(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = Interface(api=None)
        self.session_dir = Path("sessions/test_preview_session")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.ui.session_dir = self.session_dir

        (self.session_dir / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (self.session_dir / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (self.session_dir / "img2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (self.session_dir / "img3.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (self.session_dir / "clip0.mp4").write_bytes(b"\x00\x00\x00\x1cftyp")
        (self.session_dir / "track0.mp3").write_bytes(b"ID3")
        (self.session_dir / "track1.mp3").write_bytes(b"ID3")
        (self.session_dir / "track2.mp3").write_bytes(b"ID3")
        (self.session_dir / "track3.mp3").write_bytes(b"ID3")
        (self.session_dir / "note.txt").write_text("hello <b>world</b>", encoding="utf-8")
        (self.session_dir / "prompt.txt").write_text(
            "line1\n<script>alert(1)</script>", encoding="utf-8"
        )

    def test_images_preview_shows_first_three_and_ellipsis(self) -> None:
        html_out = self.ui._format_images_block(
            ["img0.png", "img1.png", "img2.png", "img3.png"]
        )
        self.assertIn("Images [4]:", html_out)
        self.assertIn("...", html_out)
        self.assertEqual(html_out.count('class="result-thumb"'), 3)
        self.assertEqual(html_out.count('class="result-media"'), 4)
        self.assertIn("data-images=", html_out)
        self.assertEqual(html_out.count("gallery-img"), 7)

    def test_videos_preview_uses_video_tags(self) -> None:
        html_out = self.ui._format_videos_block(["clip0.mp4"])
        self.assertIn("Videos [1]:", html_out)
        self.assertIn("<video", html_out)
        self.assertIn('controls preload="metadata"', html_out)

    def test_sounds_preview_shows_first_three_and_ellipsis(self) -> None:
        html_out = self.ui._format_sounds_block(
            ["track0.mp3", "track1.mp3", "track2.mp3", "track3.mp3"]
        )
        self.assertIn("Sounds [4]:", html_out)
        self.assertIn("...", html_out)
        self.assertEqual(html_out.count("<audio"), 7)  # 3 preview + 4 full list
        self.assertIn('controls preload="metadata"', html_out)

    def test_text_and_prompt_escape_html(self) -> None:
        text_html = self.ui._format_text_item("Text1", "note.txt")
        prompt_html = self.ui._format_text_item("Prompt1", "prompt.txt")

        self.assertIn("hello &lt;b&gt;world&lt;/b&gt;", text_html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", prompt_html)
        self.assertNotIn("<script>", prompt_html)
        self.assertNotIn("<b>world</b>", text_html)

    def test_text_preview_truncates_to_two_hundred_chars(self) -> None:
        long_path = self.session_dir / "long.txt"
        long_path.write_text("x" * 250, encoding="utf-8")
        html_out = self.ui._format_text_item("Text1", "long.txt")
        self.assertIn("x" * 200, html_out)
        self.assertIn("…", html_out)


if __name__ == "__main__":
    unittest.main()
