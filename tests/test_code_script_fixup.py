"""Tests for generated .ah script fixups."""

from __future__ import annotations

import unittest

from externals.code.script_fixup import ensure_run_line, fixup_generated_ah


class TestCodeScriptFixup(unittest.TestCase):
    def test_unchanged_when_run_present(self) -> None:
        script = "@answer: $llm\nhi\nrun @answer\n"
        out, changed = ensure_run_line(script)
        self.assertFalse(changed)
        self.assertIn("run @answer", out)

    def test_appends_run_answer(self) -> None:
        script = "@answer: $llm\nhi\n"
        out, changed = ensure_run_line(script)
        self.assertTrue(changed)
        self.assertTrue(out.rstrip().endswith("run @answer"))

    def test_fixup_notes(self) -> None:
        out, notes = fixup_generated_ah("@main: $pass\n")
        self.assertTrue(out.endswith("run @main\n"))
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
