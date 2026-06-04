"""Tests for $model_ah_* fine-tune externals."""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.model_ah_create_jsonl.run import run as run_create_jsonl
from externals.model_ah_train_lora.run import run as run_train_lora
from externals.model_ah_merge_lora.run import run as run_merge_lora
from externals.model_ah.dataset import load_jsonl
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir

_SAMPLE_AH = """\
# Request: Write a JavaScript snippet to retry an HTTP GET with timeout.

@answer: $code
Write a JavaScript snippet to retry an HTTP GET with timeout.
Return only the final result without comments, clarifications, or explanations.

run @answer
"""


class TestModelAhExternals(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]
        self._tmpdir = tempfile.TemporaryDirectory()
        session_dir = create_session_dir(Path(self._tmpdir.name) / "sessions")
        self.session = Session(session_dir)
        self.ctx = ExternalContext(
            session=self.session,
            op_dir=self.session.next_op_dir("model_ah_test"),
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _bundle_with_ah(self, text: str = _SAMPLE_AH) -> ArrayBundle:
        bundle = ArrayBundle()
        bundle.files.append(
            self.ctx.new_link("files", ".ah", text),
        )
        return bundle

    def test_parse_actions(self) -> None:
        for name in (
            "model_ah_create_jsonl",
            "model_ah_train_lora",
            "model_ah_merge_lora",
        ):
            expr = parse_actions(f"${name}()")
            self.assertIsInstance(expr, ExternalAction)
            self.assertEqual(expr.name, name)

    def test_create_jsonl_from_files(self) -> None:
        inp = ExternalInput(bundle=self._bundle_with_ah(), args={}, prompt_text="")
        out = run_create_jsonl(self.ctx, inp)
        self.assertEqual(len(out.files), 1)
        rows = load_jsonl(Path(self.ctx.base_dir) / out.files[0])
        self.assertEqual(len(rows), 1)
        user = rows[0]["messages"][0]["content"]
        self.assertIn("JavaScript snippet", user)
        self.assertNotIn("Write an Anthill", user)
        self.assertIn("run @answer", rows[0]["messages"][1]["content"])

    def test_create_jsonl_request_prefix(self) -> None:
        prefix = "Write an Anthill (.ah) script for this request:"
        inp = ExternalInput(
            bundle=self._bundle_with_ah(),
            args={"request_prefix": prefix},
            prompt_text="",
        )
        out = run_create_jsonl(self.ctx, inp)
        rows = load_jsonl(Path(self.ctx.base_dir) / out.files[0])
        user = rows[0]["messages"][0]["content"]
        self.assertTrue(user.startswith(prefix))
        self.assertIn("JavaScript snippet", user)

    def test_train_lora_emulated(self) -> None:
        os.environ["AH_EMULATE_MODEL_AH_TRAIN_LORA"] = "1"
        try:
            create_out = run_create_jsonl(
                self.ctx,
                ExternalInput(bundle=self._bundle_with_ah(), args={}, prompt_text=""),
            )
            inp = ExternalInput(
                bundle=ArrayBundle(files=list(create_out.files)),
                args={"epochs": "1"},
                prompt_text="",
            )
            out = run_train_lora(self.ctx, inp)
            self.assertEqual(len(out.files), 1)
            zip_path = Path(self.ctx.base_dir) / out.files[0]
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
            self.assertIn("adapter_config.json", names)
        finally:
            os.environ.pop("AH_EMULATE_MODEL_AH_TRAIN_LORA", None)

    def test_merge_lora_emulated(self) -> None:
        os.environ["AH_EMULATE_MODEL_AH_TRAIN_LORA"] = "1"
        os.environ["AH_EMULATE_MODEL_AH_MERGE_LORA"] = "1"
        try:
            create_out = run_create_jsonl(
                self.ctx,
                ExternalInput(bundle=self._bundle_with_ah(), args={}, prompt_text=""),
            )
            train_out = run_train_lora(
                self.ctx,
                ExternalInput(
                    bundle=ArrayBundle(files=list(create_out.files)),
                    args={},
                    prompt_text="",
                ),
            )
            merge_out = run_merge_lora(
                self.ctx,
                ExternalInput(
                    bundle=ArrayBundle(files=list(train_out.files)),
                    args={"quant": "Q4_K_M"},
                    prompt_text="",
                ),
            )
            self.assertEqual(len(merge_out.files), 1)
            gguf = Path(self.ctx.base_dir) / merge_out.files[0]
            self.assertTrue(gguf.is_file())
            self.assertTrue(gguf.read_bytes().startswith(b"GGUF"))
        finally:
            os.environ.pop("AH_EMULATE_MODEL_AH_TRAIN_LORA", None)
            os.environ.pop("AH_EMULATE_MODEL_AH_MERGE_LORA", None)


if __name__ == "__main__":
    unittest.main()
