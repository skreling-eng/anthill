"""Regression: multi-prompt pipeline must stay separate through @blond_woman."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import Runtime, Session, create_session_dir


class TestSimpleImageGenerationPrompts(unittest.TestCase):
    def test_blond_woman_keeps_three_prompts(self) -> None:
        source = Path("examples/example_simple_image_generation.ah").read_text(
            encoding="utf-8"
        )
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "llm,image,texts_to_prompts,list"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("rand_prompt")
        self.assertEqual(len(result.prompts), 3)


if __name__ == "__main__":
    unittest.main()
