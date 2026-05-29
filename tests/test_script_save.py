"""Tests for default script save/load in app.py."""

from __future__ import annotations

import unittest
from pathlib import Path

from app import (
    DEFAULT_SCRIPT,
    LinkApi,
    Interface,
    actions_save_path,
    load_saved_actions,
    load_saved_script,
    save_actions,
    save_script,
    script_save_path,
)


class TestScriptSave(unittest.TestCase):
    def test_load_returns_default_when_missing(self) -> None:
        base_dir = Path("sessions/_script_save_test_missing")
        self.assertEqual(load_saved_script(base_dir), DEFAULT_SCRIPT)

    def test_script_buffer_used_when_js_unavailable(self) -> None:
        base_dir = Path("sessions/_script_save_test_buffer")
        api = LinkApi(base_dir)
        api.update_script_text("@run:\n  @hello\n\nrun @run\n")
        self.assertEqual(api.script_text_for_save(), "@run:\n  @hello\n\nrun @run\n")

    def test_save_script_now_writes_file(self) -> None:
        base_dir = Path("sessions/_script_save_test_now")
        api = LinkApi(base_dir)
        api.update_script_text("@demo:\n  x\n")
        path = api.save_script_now()
        self.assertTrue(Path(path).is_file())
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "@demo:\n  x\n")

    def test_save_and_load_roundtrip(self) -> None:
        base_dir = Path("sessions/_script_save_test_roundtrip")
        source = "@demo:\nhello\n\nrun @demo\n"
        save_script(source, base_dir)
        path = script_save_path(base_dir)
        self.assertTrue(path.is_file())
        self.assertEqual(load_saved_script(base_dir), source)


class TestActionsSave(unittest.TestCase):
    def test_actions_roundtrip(self) -> None:
        base_dir = Path("sessions/_actions_save_test")
        entries = [
            {"tm": "2026-01-01 12:00:00", "data": "<b>START</b> @run"},
            {"tm": "2026-01-01 12:00:01", "data": "<b>FINISH</b> @run"},
        ]
        save_actions(entries, base_dir)
        self.assertTrue(actions_save_path(base_dir).is_file())
        self.assertEqual(load_saved_actions(base_dir), entries)

    def test_interface_load_actions(self) -> None:
        base_dir = Path("sessions/_actions_load_test")
        save_actions([{"tm": "t1", "data": "<b>Run started</b>"}], base_dir)
        api = LinkApi(base_dir)
        ui = Interface(api)
        ui.load_actions_from_disk()
        self.assertEqual(len(ui.data), 1)
        self.assertIn("Run started", ui.data[0]["data"])


if __name__ == "__main__":
    unittest.main()
