"""Tests for $math external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.math.model_paths import HF_GGUF, HF_REPO, MODEL_GGUF
from externals.math.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestMathParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$math")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "math")

    def test_parse_model(self) -> None:
        expr = parse_actions("$math(model='qwen36', max_tokens=256)")
        self.assertEqual(expr.args.get("model"), "qwen36")


class TestMathPaths(unittest.TestCase):
    def test_upstream_gguf_name(self) -> None:
        self.assertEqual(HF_REPO, "unsloth/Qwen3.6-35B-A3B-GGUF")
        self.assertEqual(HF_GGUF, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        self.assertIn("Qwen3.6", str(MODEL_GGUF))


class TestMathEmulate(unittest.TestCase):
    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_MATH"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("math"))
            bundle = ArrayBundle()
            bundle.prompts.append(
                ctx.new_link("prompts", ".txt", "Solve 2+2.\n")
            )
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            text = ctx.read_link_text(out.texts[0])
            self.assertIn("[emulated $math", text)
            self.assertIn("mathematics", text)
        finally:
            os.environ.pop("AH_EMULATE_MATH", None)


if __name__ == "__main__":
    unittest.main()
