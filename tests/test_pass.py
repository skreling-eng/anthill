"""Tests for $pass external."""

from __future__ import annotations

import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
import importlib

run = importlib.import_module("externals.pass.run").run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestPass(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$pass")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "pass")

    def test_returns_copy_of_input(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("pass")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        bundle.prompts.append(ctx.new_link("prompts", ".txt", "hello\n"))
        bundle.texts.append(ctx.new_link("texts", ".txt", "context\n"))
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        out = run(ctx, inp)
        self.assertEqual(out.prompts, bundle.prompts)
        self.assertEqual(out.texts, bundle.texts)
        self.assertIsNot(out, bundle)


if __name__ == "__main__":
    unittest.main()
