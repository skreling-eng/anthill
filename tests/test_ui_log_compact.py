"""Tests for UI log compact mode (avoid freezing on large bundles)."""

from __future__ import annotations

import unittest
from pathlib import Path

from app import Interface, LinkApi


class TestUiLogCompact(unittest.TestCase):
    def test_compact_finish_when_many_links(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        output = {
            "videos": [f"D:/v{i}.mkv" for i in range(10)],
            "texts": [f"D:/s{i}.ass" for i in range(10)],
        }
        ui.action_finish("@data", output)
        self.assertEqual(len(ui.data), 1)
        entry = ui.data[0]
        self.assertNotIn("finish_preview", entry)
        self.assertIn("videos [10]", entry["data"])
        self.assertIn("texts [10]", entry["data"])

    def test_ass_text_not_inlined(self) -> None:
        api = LinkApi(Path(".").resolve())
        ui = Interface(api)
        repo = Path(__file__).resolve().parents[1]
        ass = repo / "test_ui_log_tmp.ass"
        ass.write_text("[Script Info]\nTitle: x\n", encoding="utf-8")
        try:
            html = ui._format_text_item("Text1", str(ass.resolve()))
            self.assertIn("omitted", html)
            self.assertNotIn("[Script Info]", html)
        finally:
            ass.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
