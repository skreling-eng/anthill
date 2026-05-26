"""Tests for $only external."""

from __future__ import annotations

import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.only.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestOnlyParse(unittest.TestCase):
    def test_parse_only_arrays(self) -> None:
        expr = parse_actions("$only(images, prompts)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "only")
        self.assertEqual(expr.args.get("_arrays"), "images, prompts")


class TestOnlyRuntime(unittest.TestCase):
    def test_keeps_only_listed_arrays(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("only")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(
            prompts=[session.new_link(op_dir, "prompts", ".txt", "p\n")],
            texts=[session.new_link(op_dir, "texts", ".txt", "t\n")],
            images=[session.new_link(op_dir, "images", ".png", b"png")],
            sounds=["sounds/x.mp3"],
        )
        inp = ExternalInput(
            bundle=bundle,
            args={"_arrays": "images, prompts"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.images), 1)
        self.assertEqual(len(out.prompts), 1)
        self.assertEqual(out.texts, [])
        self.assertEqual(out.sounds, [])
        self.assertEqual(out.changes, [])

    def test_integration_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        source = """
@p
prompt

@t
text

@run: @p -> @t -> $only(prompts)
"""
        result, _ = _run(source, "run")
        self.assertEqual(len(result.prompts), 1)
        self.assertEqual(result.texts, [])


if __name__ == "__main__":
    unittest.main()
