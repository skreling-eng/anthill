"""Tests for ^ callback actions in .ah pipelines."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import CallbackAction, SequenceAction, parse_actions
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir
from externals.api import ExternalInput


class RecordingChatCallback:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.starts: list[str] = []
        self.finishes: list[str] = []

    def ah_action(
        self,
        name: str,
        bundle: ArrayBundle,
        inp: ExternalInput,
        args: dict[str, str],
        repeat: int = 1,
    ) -> ArrayBundle:
        self.calls.append(name)
        out = bundle.copy()
        out.texts.append(f"^{name}")
        return out

    def action_start(self, action_name: str) -> None:
        self.starts.append(action_name)

    def action_finish(
        self,
        action_name: str,
        output_context: dict,
        output_json_path: str | None = None,
        session_base_dir: str | None = None,
    ) -> None:
        self.finishes.append(action_name)

    def action_error(self, action_name: str, error_message: str) -> None:
        raise AssertionError(f"unexpected error {action_name}: {error_message}")


class TestCallbackParse(unittest.TestCase):
    def test_parse_simple(self) -> None:
        expr = parse_actions("^user_input -> ^store_message")
        self.assertIsInstance(expr, SequenceAction)
        self.assertIsInstance(expr.steps[0], CallbackAction)
        self.assertEqual(expr.steps[0].name, "user_input")
        self.assertIsInstance(expr.steps[1], CallbackAction)
        self.assertEqual(expr.steps[1].name, "store_message")

    def test_parse_with_args(self) -> None:
        expr = parse_actions("^answer(mode='brief')")
        self.assertIsInstance(expr, CallbackAction)
        self.assertEqual(expr.name, "answer")
        self.assertEqual(expr.args.get("mode"), "brief")


class TestCallbackRuntime(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "clear"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def test_runtime_invokes_ah_action(self) -> None:
        source = """
@run: ^ping -> ^pong

run @run
"""
        cb = RecordingChatCallback()
        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        runtime = Runtime(program, Session(session_dir), callback=cb)
        result = runtime.run()
        self.assertEqual(cb.calls, ["ping", "pong"])
        self.assertEqual(result.texts, ["^ping", "^pong"])
        self.assertIn("^ping", cb.starts)
        self.assertIn("^pong", cb.finishes)


if __name__ == "__main__":
    unittest.main()
