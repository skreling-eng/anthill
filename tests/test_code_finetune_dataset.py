"""Tests for Anthill codegen fine-tune dataset export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.code_finetune_dataset import build_dataset, write_jsonl


class TestCodeFinetuneDataset(unittest.TestCase):
    def test_export_numbered_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example_greeting_1.ah").write_text(
                "# Request: hi\n\n@answer: $llm\nhi\nReturn only the final result.\n\nrun @answer\n",
                encoding="utf-8",
            )
            rows = build_dataset([root])
            self.assertEqual(len(rows), 1)
            self.assertIn("hi", rows[0]["messages"][0]["content"])
            self.assertIn("run @answer", rows[0]["messages"][1]["content"])
            self.assertEqual(rows[0]["messages"][0]["role"], "user")
            self.assertEqual(rows[0]["messages"][1]["role"], "assistant")

    def test_skips_template_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example_llm.ah").write_text(
                "@answer: $llm\n<prompt>\nrun @answer\n",
                encoding="utf-8",
            )
            self.assertEqual(build_dataset([root]), [])

    def test_write_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train.jsonl"
            write_jsonl([{"messages": [{"role": "user", "content": "x"}]}], out)
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
