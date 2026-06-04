"""Tests for $ah_code_examples external."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.ah_code_examples.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestAhCodeExamples(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]
        session_dir = create_session_dir(self.repo / "sessions")
        self.session = Session(session_dir)
        op_dir = self.session.next_op_dir("ah_code_examples")
        self.ctx = ExternalContext(session=self.session, op_dir=op_dir)

    def _run(self, folder: str, *, per_usecase: str = "20") -> dict:
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"folder": folder, "per_usecase": per_usecase},
            prompt_text="",
        )
        out = run(self.ctx, inp)
        self.assertEqual(len(out.texts), 1)
        raw = self.ctx.read_link_text(out.texts[0])
        return json.loads(raw)

    def test_parse_action(self) -> None:
        expr = parse_actions(
            "$ah_code_examples(folder='test_data/examples', per_usecase=5)"
        )
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "ah_code_examples")
        self.assertEqual(expr.args.get("folder"), "test_data/examples")
        self.assertEqual(expr.args.get("per_usecase"), "5")

    def test_caps_per_usecase_and_parses_request(self) -> None:
        tmp = self.repo / "sessions" / "_ah_code_examples_test"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "example_alpha_2.ah").write_text(
            "# Request: second alpha\n@x: $pass\nrun @x\n",
            encoding="utf-8",
        )
        (tmp / "example_alpha_1.ah").write_text(
            "# Request: first alpha\n@x: $pass\nrun @x\n",
            encoding="utf-8",
        )
        (tmp / "example_alpha_3.ah").write_text(
            "# Request: third alpha\n@x: $pass\nrun @x\n",
            encoding="utf-8",
        )
        (tmp / "ignore.ah").write_text("not an example\n", encoding="utf-8")
        try:
            data = self._run(str(tmp.relative_to(self.repo)), per_usecase="2")
            alpha = data["usecases"]["alpha"]
            self.assertEqual(len(alpha), 2)
            self.assertEqual(alpha[0]["request"], "first alpha")
            self.assertEqual(alpha[1]["request"], "second alpha")
            self.assertEqual(data["per_usecase"], 2)
        finally:
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()

    def test_real_folder_respects_limit(self) -> None:
        data = self._run("test_data/examples", per_usecase="3")
        self.assertLessEqual(len(data["usecases"]["greeting"]), 3)
        self.assertIn("folder", data)
        self.assertTrue(data["usecases"])


if __name__ == "__main__":
    unittest.main()
