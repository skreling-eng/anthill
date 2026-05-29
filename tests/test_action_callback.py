"""Tests for optional Runtime action progress callbacks."""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir


@dataclass
class RecordingCallback:
    starts: list[str] = field(default_factory=list)
    finishes: list[tuple[str, dict]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def action_start(self, action_name: str) -> None:
        self.starts.append(action_name)

    def action_finish(
        self,
        action_name: str,
        output_context: dict,
        output_json_path: str | None = None,
    ) -> None:
        self.finishes.append((action_name, output_context, output_json_path))

    def action_error(self, action_name: str, error_message: str) -> None:
        self.errors.append((action_name, error_message))


class TestActionCallback(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "clear,list"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def _runtime(self, source: str, callback: RecordingCallback) -> Runtime:
        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        return Runtime(program, Session(session_dir), callback=callback)

    def test_instruction_and_external_callbacks(self) -> None:
        source = """
@hello:
world

@run: @hello -> $clear

run @run
"""
        callback = RecordingCallback()
        runtime = self._runtime(source, callback)
        runtime.run()

        self.assertEqual(
            callback.starts,
            ["@run", "@hello", "$clear"],
        )
        self.assertEqual(
            [name for name, *_ in callback.finishes], ["@hello", "$clear", "@run"]
        )
        self.assertEqual(callback.errors, [])

    def test_context_action_callbacks(self) -> None:
        source = """
@seed:
seed text

@run: @seed -> %track -> track%

run @run
"""
        callback = RecordingCallback()
        runtime = self._runtime(source, callback)
        runtime.run()

        self.assertIn("%track", callback.starts)
        self.assertIn("track%", callback.starts)
        self.assertIn("%track", [name for name, *_ in callback.finishes])
        self.assertIn("track%", [name for name, *_ in callback.finishes])

    def test_action_error_on_unknown_instruction(self) -> None:
        source = """
@run: @missing

run @run
"""
        callback = RecordingCallback()
        runtime = self._runtime(source, callback)

        with self.assertRaises(KeyError):
            runtime.run()

        self.assertEqual(callback.starts, ["@run", "@missing"])
        self.assertEqual(callback.finishes, [])
        self.assertEqual(len(callback.errors), 2)
        self.assertEqual(callback.errors[0][0], "@missing")
        self.assertEqual(callback.errors[1][0], "@run")
        self.assertIn("Unknown instruction", callback.errors[0][1])

    def test_internal_instruction_resolution_is_silent(self) -> None:
        source = """
@value:
hello

@run: $list

run @run
"""
        callback = RecordingCallback()
        runtime = self._runtime(source, callback)
        runtime.run()

        self.assertIn("@run", callback.starts)
        self.assertIn("$list", callback.starts)
        self.assertNotIn("@value", callback.starts)

    def test_no_callback_has_no_overhead(self) -> None:
        source = """
@hello:
ok

@run: @hello

run @run
"""
        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        runtime = Runtime(program, Session(session_dir))
        result = runtime.run()
        self.assertIsInstance(result, ArrayBundle)


if __name__ == "__main__":
    unittest.main()
