"""Tests for $json2texts external."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.json2texts.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestJson2TextsParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$json2texts(field='text')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "json2texts")
        self.assertEqual(expr.args.get("field"), "text")


class TestJson2TextsRun(unittest.TestCase):
    def _ctx_inp(self, payload: object) -> tuple[ExternalContext, ExternalInput]:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("json2texts")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        bundle.texts.append(ctx.new_link("texts", ".txt", raw))
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        return ctx, inp

    def test_splits_search_results(self) -> None:
        ctx, inp = self._ctx_inp(
            [
                {"url": "https://a.example", "text": "first"},
                {"url": "https://b.example", "text": "second"},
            ]
        )
        input_links = set(inp.bundle.texts)
        out = run(ctx, inp)
        self.assertEqual(len(out.texts), 2)
        self.assertTrue(set(out.texts).isdisjoint(input_links))
        self.assertEqual(ctx.read_link_text(out.texts[0]).strip(), "first")
        self.assertEqual(ctx.read_link_text(out.texts[1]).strip(), "second")

    def test_string_elements(self) -> None:
        ctx, inp = self._ctx_inp(["alpha", "beta"])
        out = run(ctx, inp)
        self.assertEqual(
            [ctx.read_link_text(l).strip() for l in out.texts],
            ["alpha", "beta"],
        )

    def test_join(self) -> None:
        ctx, inp = self._ctx_inp([{"text": "one"}, {"text": "two"}])
        inp.args["join"] = "1"
        out = run(ctx, inp)
        self.assertEqual(len(out.texts), 1)
        self.assertEqual(ctx.read_link_text(out.texts[0]).strip(), "one\n\ntwo")

    def test_envelope_results_key(self) -> None:
        ctx, inp = self._ctx_inp({"results": [{"text": "nested"}]})
        out = run(ctx, inp)
        self.assertEqual(ctx.read_link_text(out.texts[0]).strip(), "nested")


if __name__ == "__main__":
    unittest.main()
