"""Tests for $add_gguf_llm_model and llm_user resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.add_gguf_llm_model.run import run
from externals.llm.model_list import get_llm
from externals.llm.user_models import (
    parse_gguf_source,
    sanitize_model_name,
    user_gguf_path,
)
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestSanitize(unittest.TestCase):
    def test_model_name(self) -> None:
        self.assertEqual(sanitize_model_name("my-model_01"), "my-model_01")
        self.assertEqual(sanitize_model_name("bad name!"), "badname")

    def test_reject_empty(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_model_name("!!!")


class TestParseGguf(unittest.TestCase):
    def test_hf_resolve_url(self) -> None:
        kind, repo, fname = parse_gguf_source(
            "https://huggingface.co/unsloth/Foo/resolve/main/Bar-Q4_K_M.gguf"
        )
        self.assertEqual(kind, "hf")
        self.assertEqual(repo, "unsloth/Foo")
        self.assertEqual(fname, "Bar-Q4_K_M.gguf")

    def test_hf_shorthand(self) -> None:
        kind, repo, fname = parse_gguf_source("unsloth/Foo::Bar-Q4_K_M.gguf")
        self.assertEqual((kind, repo, fname), ("hf", "unsloth/Foo", "Bar-Q4_K_M.gguf"))

    def test_hf_shorthand_preserves_dots(self) -> None:
        kind, repo, fname = parse_gguf_source(
            "unsloth/Qwen3.6-35B-A3B-GGUF::Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
        )
        self.assertEqual(kind, "hf")
        self.assertEqual(repo, "unsloth/Qwen3.6-35B-A3B-GGUF")
        self.assertEqual(fname, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

    def test_hf_resolve_url_preserves_dots(self) -> None:
        kind, repo, fname = parse_gguf_source(
            "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/"
            "resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf?download=true"
        )
        self.assertEqual(kind, "hf")
        self.assertEqual(repo, "unsloth/Qwen3.6-35B-A3B-GGUF")
        self.assertEqual(fname, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


class TestAddGgufRun(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions(
            "$add_gguf_llm_model(name='qwen36', gguf='unsloth/X::f.gguf')"
        )
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "add_gguf_llm_model")

    def test_passthrough_and_emulate(self) -> None:
        os.environ["AH_EMULATE_ADD_GGUF_LLM_MODEL"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                models = Path(tmp) / "models"
                models.mkdir()
                with mock.patch(
                    "externals.llm.user_models.models_roots",
                    return_value=(models,),
                ):
                    session_dir = create_session_dir(Path("sessions"))
                    session = Session(session_dir)
                    ctx = ExternalContext(
                        session=session,
                        op_dir=session.next_op_dir("add_gguf"),
                    )
                    bundle = ArrayBundle()
                    bundle.texts.append(ctx.new_link("texts", ".txt", "keep\n"))
                    inp = ExternalInput(
                        bundle=bundle,
                        args={
                            "name": "test_model",
                            "gguf": "unsloth/Foo::Bar.gguf",
                        },
                        prompt_text="",
                    )
                    out = run(ctx, inp)
                    self.assertEqual(out.texts, bundle.texts)
                    path = user_gguf_path("test_model")
                    self.assertIsNotNone(path)
                    self.assertTrue(path.is_file())
        finally:
            os.environ.pop("AH_EMULATE_ADD_GGUF_LLM_MODEL", None)

    def test_get_llm_user_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            dest = models / "llm_user" / "mymodel" / "weights.gguf"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"GGUF")
            with mock.patch(
                "externals.llm.user_models.models_roots",
                return_value=(models,),
            ):
                llm = get_llm("mymodel")
                self.assertIn("weights.gguf", llm.gguf_path)


if __name__ == "__main__":
    unittest.main()
