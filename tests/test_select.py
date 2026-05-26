"""Tests for $select external."""

from __future__ import annotations

import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.select.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestSelectParse(unittest.TestCase):
    def test_parse_single_array(self) -> None:
        expr = parse_actions("$select(sounds=[1])")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "select")
        self.assertEqual(expr.args.get("sounds"), "[1]")

    def test_parse_multiple_arrays(self) -> None:
        expr = parse_actions("$select(text=[2], sounds=[0, 1, 2])")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.args.get("text"), "[2]")
        self.assertEqual(expr.args.get("sounds"), "[0, 1, 2]")


class TestSelectRuntime(unittest.TestCase):
    def test_selects_by_index(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("select")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(
            texts=[
                session.new_link(op_dir, "texts", ".txt", "a\n"),
                session.new_link(op_dir, "texts", ".txt", "b\n"),
                session.new_link(op_dir, "texts", ".txt", "c\n"),
            ],
            sounds=["sounds/v0.wav", "sounds/v1.wav", "sounds/v2.wav"],
            images=["images/x.png"],
        )
        inp = ExternalInput(
            bundle=bundle,
            args={"text": "[2]", "sounds": "[0, 1, 2]"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(out.texts, [bundle.texts[2]])
        self.assertEqual(out.sounds, bundle.sounds[:3])
        self.assertEqual(out.images, [])
        self.assertEqual(out.prompts, [])

    def test_selects_single_sound(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("select")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/a.wav", "sounds/b.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={"sounds": "[1]"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(out.sounds, ["sounds/b.wav"])

    def test_arg_list_indexes(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("select")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/a.wav", "sounds/b.wav", "sounds/c.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={},
            prompt_text="",
            arg_lists={"sounds": ["0", "2"]},
        )
        out = run(ctx, inp)
        self.assertEqual(out.sounds, ["sounds/a.wav", "sounds/c.wav"])

    def test_out_of_range(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("select")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle(sounds=["sounds/a.wav"])
        inp = ExternalInput(
            bundle=bundle,
            args={"sounds": "[3]"},
            prompt_text="",
        )
        with self.assertRaises(IndexError):
            run(ctx, inp)

    def test_requires_specs(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("select")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(ValueError):
            run(ctx, inp)

    def test_integration_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        source = """
@p
a, b, c

@run: @p -> $list -> $select(texts=[2])
"""
        result, _ = _run(source, "run", inprocess="list,select")
        self.assertEqual(len(result.texts), 1)
        self.assertEqual(result.prompts, [])


if __name__ == "__main__":
    unittest.main()
