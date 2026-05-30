"""Tests for $llm external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.llm.run import _build_prompt, run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestLlmParse(unittest.TestCase):
    def test_parse_add_texts(self) -> None:
        expr = parse_actions("$llm(add_texts=True)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "llm")
        self.assertEqual(expr.args.get("add_texts"), "True")


class TestLlmPrompt(unittest.TestCase):
    def _ctx_inp(
        self,
        *,
        prompt_text: str = "",
        texts: list[str] | None = None,
        add_texts: bool = False,
    ) -> tuple[ExternalContext, ExternalInput]:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("llm")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        for text in texts or []:
            bundle.texts.append(ctx.new_link("texts", ".txt", text))
        args = {"add_texts": "True"} if add_texts else {}
        inp = ExternalInput(bundle=bundle, args=args, prompt_text=prompt_text)
        return ctx, inp

    def test_default_merges_texts_inline(self) -> None:
        ctx, inp = self._ctx_inp(
            prompt_text="Summarize this",
            texts=["line one", "line two"],
        )
        prompt = _build_prompt(inp, ctx)
        self.assertEqual(prompt, "Summarize this\n\nline one\n\nline two")
        self.assertNotIn("__CONTENT__", prompt)

    def test_add_texts_puts_content_in_section(self) -> None:
        ctx, inp = self._ctx_inp(
            prompt_text="Format this output",
            texts=["raw output", "more output"],
            add_texts=True,
        )
        prompt = _build_prompt(inp, ctx)
        self.assertEqual(
            prompt,
            "Format this output\n\n__CONTENT__\nraw output\n\nmore output",
        )

    def test_add_texts_without_prompt(self) -> None:
        ctx, inp = self._ctx_inp(texts=["only content"], add_texts=True)
        prompt = _build_prompt(inp, ctx)
        self.assertEqual(prompt, "__CONTENT__\nonly content")

    def test_emulate_includes_content_section(self) -> None:
        os.environ["AH_EMULATE_LLM"] = "1"
        try:
            ctx, inp = self._ctx_inp(
                prompt_text="Format this",
                texts=["code result"],
                add_texts=True,
            )
            out = run(ctx, inp)
            text = ctx.read_link_text(out.texts[-1])
            self.assertIn("__CONTENT__", text)
            self.assertIn("code result", text)
        finally:
            os.environ.pop("AH_EMULATE_LLM", None)

    def test_output_omits_input_texts(self) -> None:
        os.environ["AH_EMULATE_LLM"] = "1"
        try:
            ctx, inp = self._ctx_inp(
                prompt_text="Summarize",
                texts=["context one", "context two"],
            )
            input_links = set(inp.bundle.texts)
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            self.assertTrue(set(out.texts).isdisjoint(input_links))
        finally:
            os.environ.pop("AH_EMULATE_LLM", None)


if __name__ == "__main__":
    unittest.main()
