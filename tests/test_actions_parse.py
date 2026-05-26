"""Tests for .ah action expression parsing."""

from __future__ import annotations

import unittest

from ahlib.ah_actions import (
    ExternalAction,
    ParallelAction,
    RefAction,
    SequenceAction,
    _tokenize_actions,
    parse_actions,
)
from ahlib.ah_parser import parse_ah_file


class TestParallelComfyParse(unittest.TestCase):
    """Regression: parallel ( $comfy(...), $comfy(...) ) must not swallow closing )."""

    ACTIONS = (
        "( @image ) -> ( $comfy(port=8000, json='Qwen-Rapid-AIO_4.json'), "
        "$comfy(port=8000, json='Rapid-AIO-Mega__3_exp.json') )"
    )

    EXPECTED_TOKENS = [
        "(",
        "@image",
        ")",
        "->",
        "(",
        "$comfy(port=8000, json='Qwen-Rapid-AIO_4.json')",
        ",",
        "$comfy(port=8000, json='Rapid-AIO-Mega__3_exp.json')",
        ")",
    ]

    def test_tokenize_exact_tokens(self) -> None:
        self.assertEqual(_tokenize_actions(self.ACTIONS), self.EXPECTED_TOKENS)

    def test_tokenize_splits_parallel_closing_paren(self) -> None:
        tokens = _tokenize_actions(self.ACTIONS)
        self.assertEqual(tokens[-1], ")")
        self.assertTrue(
            all(" )" not in t for t in tokens if t.startswith("$")),
            f"externals must not include parallel close paren: {tokens}",
        )

    def test_parse_does_not_raise_unclosed_paren(self) -> None:
        expr = parse_actions(self.ACTIONS)
        self.assertIsNotNone(expr)

    def test_parse_parallel_comfy_branches(self) -> None:
        expr = parse_actions(self.ACTIONS)
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 2)
        self.assertIsInstance(expr.steps[0], ParallelAction)
        self.assertIsInstance(expr.steps[0].branches[0], RefAction)
        self.assertEqual(expr.steps[0].branches[0].name, "image")
        par2 = expr.steps[1]
        self.assertIsInstance(par2, ParallelAction)
        self.assertEqual(len(par2.branches), 2)
        workflows = []
        for branch in par2.branches:
            self.assertIsInstance(branch, ExternalAction)
            self.assertEqual(branch.name, "comfy")
            self.assertEqual(branch.args.get("port"), "8000")
            workflows.append(branch.args.get("json"))
        self.assertEqual(
            workflows,
            ["Qwen-Rapid-AIO_4.json", "Rapid-AIO-Mega__3_exp.json"],
        )

    def test_example_comfy_realistic_instruction(self) -> None:
        inst = parse_ah_file("example_comfy.ah")["instructions"]["realistic"]
        expr = parse_actions(inst["actions"])
        self.assertIsInstance(expr, SequenceAction)
        tail = expr.steps[-1]
        self.assertIsInstance(tail, SequenceAction)
        par = tail.steps[-1]
        self.assertIsInstance(par, ParallelAction)
        self.assertEqual(len(par.branches), 2)


class TestExternalRepeatSuffix(unittest.TestCase):
    """$name(...)[n] must not break after tokenizer fix."""

    def test_external_with_repeat_suffix(self) -> None:
        tokens = _tokenize_actions("$image(model='x')[3]")
        self.assertEqual(tokens, ["$image(model='x')[3]"])
        expr = parse_actions("$image(model='x')[3]")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "image")
        self.assertEqual(expr.repeat, 3)

    def test_collapses_line_break_arrow_duplicates(self) -> None:
        tokens = _tokenize_actions("%stems ->\n-> $select(sounds=[0])")
        self.assertEqual(tokens, ["%stems", "->", "$select(sounds=[0])"])
        expr = parse_actions("%stems -> -> $select(sounds=[0])")
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 2)


if __name__ == "__main__":
    unittest.main()
