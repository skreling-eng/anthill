"""Tests for $search / $serch external."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.search.duckduckgo import (
    normalize_instant_results,
    parse_html_results,
    unwrap_url,
)
from externals.search.page_fetch import (
    enrich_results_with_pages,
    extract_text_from_html,
)
from externals.search.run import run
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


class TestSearchParse(unittest.TestCase):
    def test_parse_search(self) -> None:
        expr = parse_actions("$search(limit=5)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "search")
        self.assertEqual(expr.args.get("limit"), "5")

    def test_parse_serch_alias(self) -> None:
        expr = parse_actions("$serch(limit=3)")
        self.assertEqual(expr.name, "serch")


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


_SAMPLE_PAGE_HTML = """
<html><head><title>T</title><script>alert(1)</script></head>
<body>
<nav>Home | About</nav>
<div class="content">
  <h1>Main Article</h1>
  <p>Important fact about birds.</p>
</div>
<footer>Copyright 2026. Accept all cookies.</footer>
</body></html>
"""


_EXAMPLE_COM_HTML = """<!doctype html><html lang="en"><head><title>Example Domain</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0}</style></head><body><div><h1>Example Domain</h1>
<p>This domain is for use in documentation examples.</p></div></body></html>"""


class TestPageFetch(unittest.TestCase):
    def test_extract_handles_meta_in_head(self) -> None:
        text = extract_text_from_html(_EXAMPLE_COM_HTML)
        self.assertIn("Example Domain", text)
        self.assertIn("documentation examples", text)

    def test_fetch_example_com(self) -> None:
        try:
            from externals.search.page_fetch import fetch_page_text

            text = fetch_page_text("https://example.com/", timeout=15)
        except Exception as exc:
            self.skipTest(f"network unavailable: {exc}")
        self.assertIn("Example Domain", text)

    def test_extract_strips_nav_and_scripts(self) -> None:
        text = extract_text_from_html(_SAMPLE_PAGE_HTML)
        self.assertIn("Important fact about birds", text)
        self.assertNotIn("alert(1)", text)
        self.assertNotIn("Accept all cookies", text)

    @mock.patch("externals.search.page_fetch.fetch_page_text")
    def test_enrich_adds_page_fields(self, mock_fetch: mock.MagicMock) -> None:
        mock_fetch.return_value = "Full article body."
        rows = [{"url": "https://example.com/a", "text": "snippet"}]
        out = enrich_results_with_pages(
            rows, max_pages=2, timeout=5.0, max_chars=1000
        )
        self.assertEqual(out[0]["page_text"], "Full article body.")
        self.assertEqual(out[0]["page_fetch"], "ok")
        self.assertIn("--- page ---", out[0]["text"])
        mock_fetch.assert_called_once_with(
            "https://example.com/a", timeout=5.0, max_chars=1000
        )


class TestSearchRun(unittest.TestCase):
    def test_emulate_writes_json_texts(self) -> None:
        os.environ["AH_EMULATE_SEARCH"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("search")
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
            self.assertIn("[emulated $search]", rows[0]["text"])
        finally:
            os.environ.pop("AH_EMULATE_SEARCH", None)

    def test_emulate_fetch_pages(self) -> None:
        os.environ["AH_EMULATE_SEARCH"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("search"))
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"fetch_pages": "true"},
                prompt_text="q",
            )
            out = run(ctx, inp)
            rows = json.loads(ctx.read_link_text(out.texts[0]))
            self.assertIn("page_text", rows[0])
            self.assertEqual(rows[0]["page_fetch"], "ok")
        finally:
            os.environ.pop("AH_EMULATE_SEARCH", None)

    @mock.patch("externals.search.run.enrich_results_with_pages")
    @mock.patch("externals.search.run.search")
    def test_run_fetch_pages(
        self, mock_search: mock.MagicMock, mock_enrich: mock.MagicMock
    ) -> None:
        mock_search.return_value = [{"url": "https://x.test", "text": "hello"}]
        mock_enrich.return_value = [
            {
                "url": "https://x.test",
                "text": "hello\n\n--- page ---\nbody",
                "page_text": "body",
                "page_fetch": "ok",
            }
        ]
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(session=session, op_dir=session.next_op_dir("search"))
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"limit": "2", "fetch_pages": "1", "fetch_max": "1"},
            prompt_text="test query",
        )
        out = run(ctx, inp)
        mock_search.assert_called_once()
        mock_enrich.assert_called_once()
        rows = json.loads(ctx.read_link_text(out.texts[0]))
        self.assertEqual(rows[0]["page_text"], "body")

    @mock.patch("externals.search.run.search")
    def test_run_calls_duckduckgo(self, mock_search: mock.MagicMock) -> None:
        mock_search.return_value = [
            {"url": "https://x.test", "text": "hello"},
        ]
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("search")
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
        self.assertNotIn("page_text", rows[0])


if __name__ == "__main__":
    unittest.main()
