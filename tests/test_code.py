"""Tests for $code external."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.code.run import build_request, run
from externals.llm.context_limit import auto_n_ctx, trim_code_request
from externals.code.model_list import resolve_code_n_ctx
from ahlib.ah_runtime import ArrayBundle, Session


class CodeRequestTests(unittest.TestCase):
    def _ctx_and_inp(
        self,
        *,
        prompts: list[str] | None = None,
        texts: list[str] | None = None,
        files: list[tuple[str, str]] | None = None,
        prompt_text: str = "",
    ) -> tuple[ExternalContext, ExternalInput]:
        session_dir = Path(self._session_root)
        session = Session(session_dir)
        op_dir = session.next_op_dir("code_test")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        bundle = ArrayBundle()
        for i, text in enumerate(prompts or []):
            bundle.prompts.append(ctx.new_link("prompts", ".txt", text))
        for i, text in enumerate(texts or []):
            bundle.texts.append(ctx.new_link("texts", ".txt", text))
        for path, content in files or []:
            bundle.files.append(ctx.new_link("files", Path(path).suffix or ".txt", content))

        inp = ExternalInput(bundle=bundle, args={}, prompt_text=prompt_text)
        return ctx, inp

    def setUp(self) -> None:
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self._session_root = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_build_request_joins_sections(self) -> None:
        ctx, inp = self._ctx_and_inp(
            prompts=["Fix the bug"],
            texts=["def foo():\n    pass", "class Bar: ..."],
            files=[("src/main.py", 'print("hi")\n')],
            prompt_text="Extra instruction",
        )
        req = build_request(ctx, inp)
        self.assertEqual(req["prompts"], ["Extra instruction", "Fix the bug"])
        self.assertIn("def foo():", req["code_context"])
        self.assertIn("class Bar:", req["code_context"])
        self.assertEqual(len(req["files"]), 1)
        self.assertEqual(req["files"][0]["name"], "0.py")
        self.assertIn('print("hi")', req["files"][0]["content"])

    def test_request_json_quotes_special_chars(self) -> None:
        ctx, inp = self._ctx_and_inp(
            texts=['line with "quotes" and \\ backslash'],
            files=[("a.py", '{"key": "value"}\n')],
        )
        raw = json.dumps(build_request(ctx, inp), ensure_ascii=False)
        parsed = json.loads(raw)
        self.assertIn('"quotes"', parsed["code_context"])
        self.assertEqual(parsed["files"][0]["content"], '{"key": "value"}')

    def test_auto_n_ctx_rounds_to_power_of_two(self) -> None:
        self.assertEqual(auto_n_ctx(1000, 2048, min_ctx=4096, max_ctx=131072), 4096)
        self.assertEqual(auto_n_ctx(20_000, 2048, min_ctx=4096, max_ctx=131072), 32768)
        self.assertEqual(auto_n_ctx(31_000, 2048, min_ctx=4096, max_ctx=131072), 65536)

    def test_resolve_code_n_ctx_from_payload(self) -> None:
        small = '{"prompts":["hi"],"code_context":"","files":[]}'
        self.assertEqual(
            resolve_code_n_ctx(small, 512, explicit=None),
            4096,
        )
        self.assertEqual(resolve_code_n_ctx(small, 512, explicit=8192), 8192)

    def test_resolve_code_n_ctx_caps_at_auto_max_by_default(self) -> None:
        big = "x" * 110_000
        payload = json.dumps({"prompts": ["t"], "code_context": big, "files": []})
        self.assertEqual(resolve_code_n_ctx(payload, 2048, explicit=None), 16384)

    def test_resolve_code_n_ctx_extended_env(self) -> None:
        big = "x" * 110_000
        payload = json.dumps({"prompts": ["t"], "code_context": big, "files": []})
        os.environ["AH_CODE_EXTENDED_CTX"] = "1"
        os.environ["AH_CODE_AUTO_MAX_N_CTX"] = "131072"
        try:
            self.assertEqual(
                resolve_code_n_ctx(payload, 2048, explicit=None),
                65536,
            )
        finally:
            os.environ.pop("AH_CODE_EXTENDED_CTX", None)
            os.environ.pop("AH_CODE_AUTO_MAX_N_CTX", None)

    def test_trim_code_request_fits_budget(self) -> None:
        huge = "x" * 50_000
        request = {
            "prompts": ["task", "task"],
            "code_context": "",
            "files": [{"path": "a.py", "name": "a.py", "content": huge}],
        }
        trimmed, notes = trim_code_request(request, budget_chars=8000)
        payload = json.dumps(trimmed, ensure_ascii=False)
        self.assertLessEqual(len(payload), 8000)
        self.assertTrue(trimmed.get("_truncated"))
        self.assertTrue(any("duplicate" in n for n in notes))

    def test_emulate_run_does_not_passthrough_input_texts(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            ctx, inp = self._ctx_and_inp(
                prompts=["task"],
                texts=["old context", "more context"],
            )
            input_links = list(inp.bundle.texts)
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            for link in input_links:
                self.assertNotIn(link, out.texts)
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)

    def test_emulate_run_writes_texts(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            ctx, inp = self._ctx_and_inp(prompts=["Write tests"], texts=["context"])
            out = run(ctx, inp)
            self.assertGreaterEqual(len(out.texts), 1)
            text = ctx.read_link_text(out.texts[-1])
            self.assertIn("[emulated $code", text)
            self.assertIn("Write tests", text)
            request_path = ctx.op_dir / "request.json"
            self.assertTrue(request_path.is_file())
            saved = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["prompts"], ["Write tests"])
            self.assertEqual(saved["code_context"], "context")
            model_path = ctx.op_dir / "model.json"
            self.assertTrue(model_path.is_file())
            model_info = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(model_info["external"], "code")
            self.assertEqual(model_info["model"], "default")
            self.assertTrue(model_info["emulate"])
            self.assertEqual(model_info["emulate_reason"], "AH_EMULATE_CODE")
            self.assertEqual(model_info["max_tokens"], 2048)
            self.assertFalse(model_info["ah"])
            text = ctx.read_link_text(out.texts[-1])
            self.assertIn("expert coding assistant", text)
            self.assertNotIn("Anthill", text)
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)

    def test_emulate_ah_mode_uses_anthill_system(self) -> None:
        os.environ["AH_EMULATE_CODE"] = "1"
        try:
            ctx, inp = self._ctx_and_inp(prompts=["task"])
            inp.args["ah"] = "1"
            out = run(ctx, inp)
            text = ctx.read_link_text(out.texts[-1])
            self.assertIn("Anthill", text)
            model_info = json.loads(
                (ctx.op_dir / "model.json").read_text(encoding="utf-8")
            )
            self.assertTrue(model_info["ah"])
        finally:
            os.environ.pop("AH_EMULATE_CODE", None)


if __name__ == "__main__":
    unittest.main()
