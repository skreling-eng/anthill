"""Tests for multiline .ah instruction action parsing."""

from __future__ import annotations

import unittest

from ahlib.ah_actions import ParallelAction, RefAction, SequenceAction, parse_actions
from ahlib.ah_parser import parse_ah_source


class TestMultilineActionsParse(unittest.TestCase):
    SOURCE = """@realistic: @images -> @video_prompt -> (
    $comfy(port=8000, json='Rapid-AIO-Mega__3_start_end_image.json'),
    $comfy(port=8000, json='Rapid-AIO-Mega__3_start_image.json'),
)
make this video
"""

    def test_actions_span_lines_until_close_paren(self) -> None:
        inst = parse_ah_source(self.SOURCE).instructions["realistic"]
        self.assertIn("$comfy", inst.actions or "")
        self.assertIn("start_image.json", inst.actions or "")
        self.assertNotIn("make this video", inst.actions or "")

    def test_body_after_closed_paren(self) -> None:
        inst = parse_ah_source(self.SOURCE).instructions["realistic"]
        self.assertEqual(inst.body.strip(), "make this video")

    def test_multiline_actions_parse_to_parallel_comfy(self) -> None:
        inst = parse_ah_source(self.SOURCE).instructions["realistic"]
        expr = parse_actions(inst.actions)
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 3)
        self.assertIsInstance(expr.steps[0], RefAction)
        self.assertEqual(expr.steps[0].name, "images")
        self.assertIsInstance(expr.steps[1], RefAction)
        self.assertEqual(expr.steps[1].name, "video_prompt")
        par = expr.steps[2]
        self.assertIsInstance(par, ParallelAction)
        self.assertEqual(len(par.branches), 2)
        workflows = [b.args["json"] for b in par.branches]
        self.assertEqual(
            workflows,
            [
                "Rapid-AIO-Mega__3_start_end_image.json",
                "Rapid-AIO-Mega__3_start_image.json",
            ],
        )


class TestBracedInstructionBlock(unittest.TestCase):
    """@name: { ... } multiline blocks must not include the @header in actions."""

    SOURCE = """
@replace_voice: {
     $file('test.wav') ->
     $music_separation(model='2stem') -> %stems ->
     -> $select(sounds=[0]) -> $change_voice ->
     -> (  $select(sounds=[0]),
           stems% -> $select(sounds=[1])
        )
}

run @replace_voice
"""

    def test_braced_block_strips_header(self) -> None:
        inst = parse_ah_source(self.SOURCE).instructions["replace_voice"]
        self.assertIsNotNone(inst.actions)
        assert inst.actions is not None
        self.assertNotIn("@replace_voice", inst.actions)
        self.assertTrue(inst.actions.strip().startswith("$file"))

    def test_braced_block_parses_parallel_with_context(self) -> None:
        inst = parse_ah_source(self.SOURCE).instructions["replace_voice"]
        expr = parse_actions(inst.actions)
        self.assertIsInstance(expr, SequenceAction)

        def _find_parallel(node) -> ParallelAction | None:
            if isinstance(node, ParallelAction):
                return node
            if isinstance(node, SequenceAction):
                for step in node.steps:
                    found = _find_parallel(step)
                    if found is not None:
                        return found
            return None

        par = _find_parallel(expr)
        self.assertIsNotNone(par)
        assert par is not None
        self.assertEqual(len(par.branches), 2)


class TestCaretRunShorthand(unittest.TestCase):
    def test_caret_runs_last_instruction(self) -> None:
        program = parse_ah_source(
            """@load: $file('x.png')
@describe: @load -> $image2text
>>>
"""
        )
        self.assertEqual(program.run_target, "describe")

    def test_caret_ignores_later_run_and_caret(self) -> None:
        program = parse_ah_source(
            """@a: $clear
@b: $clear
>>>
run @a
>>>
"""
        )
        self.assertEqual(program.run_target, "b")

    def test_run_before_caret_overridden_by_caret(self) -> None:
        program = parse_ah_source(
            """@a: $clear
run @a
@b: $clear
>>>
"""
        )
        self.assertEqual(program.run_target, "b")

    def test_caret_without_prior_instruction_leaves_run_unset(self) -> None:
        program = parse_ah_source(">>>\n")
        self.assertIsNone(program.run_target)


if __name__ == "__main__":
    unittest.main()
