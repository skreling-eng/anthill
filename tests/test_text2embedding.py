"""Tests for $text2embedding external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.image2embedding.embedding_format import EMBED_DIM, unpack_siglip_embedding
from externals.text2embedding.run import _source_texts, run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestText2EmbeddingParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$text2embedding(model='224', gpu=True)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "text2embedding")
        self.assertEqual(expr.args.get("model"), "224")


class TestText2EmbeddingSource(unittest.TestCase):
    def test_prefers_texts_over_prompts(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        ctx = ExternalContext(session=session, op_dir=session.next_op_dir("t2e"))
        bundle = ArrayBundle(
            texts=[ctx.new_link("texts", ".txt", "from texts\n")],
            prompts=[ctx.new_link("prompts", ".txt", "from prompts\n")],
        )
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        self.assertEqual(_source_texts(ctx, inp), ["from texts"])


class TestText2EmbeddingRun(unittest.TestCase):
    def test_emulate_one_embedding_per_text(self) -> None:
        os.environ["AH_EMULATE_TEXT2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("text2embedding")
            )
            bundle = ArrayBundle()
            bundle.texts.append(ctx.new_link("texts", ".txt", "a red bicycle\n"))
            bundle.embeddings.append("old")
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            self.assertEqual(len(out.embeddings), 1)
            encoded = out.embeddings[0]
            self.assertIsInstance(encoded, str)
            self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)
        finally:
            os.environ.pop("AH_EMULATE_TEXT2EMBEDDING", None)

    def test_no_texts_clears_embeddings(self) -> None:
        os.environ["AH_EMULATE_TEXT2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("text2embedding")
            )
            inp = ExternalInput(
                bundle=ArrayBundle(embeddings=["deadbeef"]),
                args={},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(out.embeddings, [])
        finally:
            os.environ.pop("AH_EMULATE_TEXT2EMBEDDING", None)


if __name__ == "__main__":
    unittest.main()
