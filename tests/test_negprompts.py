"""Tests for negprompts[] array and prompt/negprompt conversion externals."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests.test_prompt_merge import _run


class TestNegpromptsExternals(unittest.TestCase):
    def test_prompts_to_negprompts_clears_prompts(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = (
            "prompts_to_negprompts,negprompts_to_prompts,prompts2negprompts,negprompts2prompts"
        )
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            source = """
@pos
sunny meadow

@run: @pos -> $prompts2negprompts -> $negprompts2prompts
"""
            result, session_dir = _run(source, "run")
            self.assertEqual(result.negprompts, [])
            self.assertEqual(len(result.prompts), 1)
            texts = [
                (session_dir / link).read_text(encoding="utf-8").strip()
                for link in result.prompts
            ]
            self.assertEqual(texts, ["sunny meadow"])
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)

    def test_prompts_to_negprompts_moves_to_negprompts(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "prompts_to_negprompts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            source = """
@bad
blurry, low quality

@run: @bad -> $prompts_to_negprompts
"""
            result, session_dir = _run(source, "run")
            self.assertEqual(result.prompts, [])
            self.assertEqual(len(result.negprompts), 1)
            text = (session_dir / result.negprompts[0]).read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(text, "blurry, low quality")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)

    def test_negprompts_context_store_and_load(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "prompts_to_negprompts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            source = """
@bad
watermark, text overlay

@run: @bad -> $prompts_to_negprompts -> %neg -> neg%
"""
            result, session_dir = _run(source, "run")
            self.assertEqual(result.prompts, [])
            self.assertEqual(len(result.negprompts), 1)
            text = (session_dir / result.negprompts[0]).read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(text, "watermark, text overlay")
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)


if __name__ == "__main__":
    unittest.main()
