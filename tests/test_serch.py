"""Tests for $serch external."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.serch.duckduckgo import (
    normalize_instant_results,
    parse_html_results,
    unwrap_url,
)
from externals.serch.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir

_SAMPLE_HTML = """
<div class="result">
  <h2><a class="result__a" href="https://example.com/a">Title A</a></h2>
  <a class="result__snippet" href="https://example.com/a">Snippet for A</a>
</div>
<div class="result">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Title B</a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Snippet B</a>
</div>
"""


class TestSerchParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$serch(limit=5)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "serch")
        self.assertEqual(expr.args.get("limit"), "5")

    def test_parse_search_alias(self) -> None:
        expr = parse_actions("$search(limit=3)")
        self.assertEqual(expr.name, "search")


class TestDuckDuckGoClient(unittest.TestCase):
    def test_parse_html(self) -> None:
        rows = parse_html_results(_SAMPLE_HTML, limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["url"], "https://example.com/a")
        self.assertIn("Title A", rows[0]["text"])
        self.assertIn("Snippet for A", rows[0]["text"])
        self.assertEqual(rows[1]["url"], "https://example.com/b")

    def test_unwrap_redirect(self) -> None:
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Ffoo.test%2Fpath"
        self.assertEqual(unwrap_url(href), "https://foo.test/path")

    def test_normalize_instant(self) -> None:
        data = {
            "Abstract": "summary",
            "AbstractURL": "https://abstract.example",
            "Results": [{"FirstURL": "https://r.example", "Text": "result text"}],
            "RelatedTopics": [
                {"FirstURL": "https://t.example", "Text": "topic text"},
            ],
        }
        rows = normalize_instant_results(data)
        self.assertEqual(rows[0]["url"], "https://abstract.example")
        self.assertEqual(rows[1]["url"], "https://r.example")
        self.assertEqual(rows[2]["url"], "https://t.example")


class TestSerchRun(unittest.TestCase):
    def test_emulate_writes_json_texts(self) -> None:
        os.environ["AH_EMULATE_SERCH"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("serch")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.prompts.append(ctx.new_link("prompts", ".txt", "anthill agent\n"))
            inp = ExternalInput(
                bundle=bundle,
                args={"limit": "3"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(out.prompts, [])
            self.assertEqual(len(out.texts), 1)
            rows = json.loads(ctx.read_link_text(out.texts[0]))
            self.assertEqual(len(rows), 1)
            self.assertIn("url", rows[0])
            self.assertIn("text", rows[0])
        finally:
            os.environ.pop("AH_EMULATE_SERCH", None)

    @mock.patch("externals.serch.run.search")
    def test_run_calls_duckduckgo(self, mock_search: mock.MagicMock) -> None:
        mock_search.return_value = [
            {"url": "https://x.test", "text": "hello"},
        ]
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("serch")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"limit": "2"},
            prompt_text="test query",
        )
        out = run(ctx, inp)
        mock_search.assert_called_once()
        self.assertEqual(mock_search.call_args.args[0], "test query")
        rows = json.loads(ctx.read_link_text(out.texts[0]))
        self.assertEqual(rows[0]["url"], "https://x.test")


if __name__ == "__main__":
    unittest.main()
