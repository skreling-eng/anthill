"""Tests for input_json copy ref formatting in app.Interface."""

from __future__ import annotations

import unittest
from pathlib import Path

from app import Interface, LinkApi, load_saved_actions, save_actions


class TestInputJsonCopy(unittest.TestCase):
    def test_input_json_ref_relative_to_project(self) -> None:
        base_dir = Path("sessions/_input_json_copy_test_proj")
        api = LinkApi(base_dir)
        ui = Interface(api)
        session_json = base_dir / "sessions" / "20260101_1_1" / "3__run" / "output.json"
        ref = ui._input_json_ref(session_json)
        self.assertEqual(
            ref,
            "input_json('sessions/20260101_1_1/3__run/output.json')",
        )

    def test_log_header_includes_copy_button(self) -> None:
        base_dir = Path("sessions/_input_json_copy_test2")
        api = LinkApi(base_dir)
        ui = Interface(api)
        ui.add_action_results(
            "<b>FINISH</b> @run",
            input_json_ref="sessions/demo/output.json",
        )
        header = ui._format_log_header(ui.data[0])
        self.assertIn("Copy the path to buffer", header)
        self.assertIn("input_json('sessions/demo/output.json')", header)

    def test_actions_save_preserves_input_json_ref(self) -> None:
        base_dir = Path("sessions/_input_json_copy_test3")
        entries = [
            {
                "tm": "t",
                "data": "<b>FINISH</b>",
                "input_json_ref": "input_json('sessions/x/output.json')",
            }
        ]
        save_actions(entries, base_dir)
        loaded = load_saved_actions(base_dir)
        self.assertEqual(loaded[0]["input_json_ref"], entries[0]["input_json_ref"])


if __name__ == "__main__":
    unittest.main()
