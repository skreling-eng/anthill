"""Tests for brain subproject (self-contained)."""

from __future__ import annotations

import os
import unittest

from brain.tools.diff import extract_diffs


class TestExternalsCatalog(unittest.TestCase):
    def test_catalog_query_detection(self) -> None:
        from brain.tools.externals_catalog import is_externals_catalog_query

        self.assertTrue(
            is_externals_catalog_query(
                "give me the list of external calls with descriptions"
            )
        )
        self.assertFalse(
            is_externals_catalog_query("add a new $search example to examples/")
        )

    def test_scan_externals(self) -> None:
        from brain.config import BrainConfig
        from brain.tools.externals_catalog import scan_externals

        root = BrainConfig().codebase_root
        entries = scan_externals(root)
        self.assertGreater(len(entries), 10)
        names = {e.name for e in entries}
        self.assertIn("search", names)
        self.assertIn("llm", names)


class TestContextLimit(unittest.TestCase):
    def test_trim_agent_prompt_drops_files(self) -> None:
        from brain.llm.context_limit import AgentPromptParts, trim_agent_prompt

        parts = AgentPromptParts(
            request="change externals",
            tree="x" * 2000,
            context_files={
                "a.py": "line\n" * 200,
                "b.py": "y" * 1500,
            },
        )
        trimmed, notes = trim_agent_prompt(parts, budget_chars=4000)
        self.assertLessEqual(trimmed.size(), 4500)
        self.assertTrue(notes)

    def test_resolve_n_ctx_auto(self) -> None:
        from brain.config import BrainConfig

        cfg = BrainConfig(n_ctx=None, auto_max_n_ctx=16384)
        small = cfg.resolve_n_ctx("short prompt", 1024)
        self.assertEqual(small, 4096)
        huge = cfg.resolve_n_ctx("x" * 50000, 4096)
        self.assertLessEqual(huge, cfg.effective_max_ctx())


class TestCodebasePaths(unittest.TestCase):
    def test_blocked_cache(self) -> None:
        from brain.tools.codebase import is_blocked_path

        self.assertTrue(is_blocked_path(".cache/llama.cpp/README.md"))
        self.assertFalse(is_blocked_path("externals/search/run.py"))


class TestDiffExtraction(unittest.TestCase):
    def test_fence_diff(self) -> None:
        text = """Analysis here.

```diff
--- a/foo.ah
+++ b/foo.ah
@@ -1,2 +1,3 @@
 @hello:
 hello
+world
```
"""
        result = extract_diffs(text)
        self.assertEqual(len(result.diffs), 1)
        self.assertIn("foo.ah", result.files_touched)

    def test_raw_diff(self) -> None:
        text = (
            "--- a/ahlib/ah_parser.py\n"
            "+++ b/ahlib/ah_parser.py\n"
            "@@ -10,3 +10,4 @@\n"
            " line\n"
            "+added\n"
        )
        result = extract_diffs(text)
        self.assertTrue(result.has_diffs)


class TestAgentEmulate(unittest.TestCase):
    def test_catalog_run_no_llm(self) -> None:
        from brain.agent.orchestrator import AgentOrchestrator
        from brain.config import BrainConfig

        cfg = BrainConfig(emulate=True)
        agent = AgentOrchestrator(cfg)
        result = agent.run("give me the list of external calls with descriptions")
        self.assertIsNone(result.error)
        self.assertEqual(result.plan.get("mode"), "externals_catalog")
        self.assertIn("externals", result.report.lower())

    def test_emulate_run(self) -> None:
        os.environ["BRAIN_EMULATE"] = "1"
        try:
            from brain.agent.orchestrator import AgentOrchestrator
            from brain.config import BrainConfig

            cfg = BrainConfig(emulate=True)
            agent = AgentOrchestrator(cfg)
            result = agent.run("Add a comment to ah_parser.py")
            self.assertIsNone(result.error)
            self.assertTrue(result.plan)
        finally:
            os.environ.pop("BRAIN_EMULATE", None)


if __name__ == "__main__":
    unittest.main()
