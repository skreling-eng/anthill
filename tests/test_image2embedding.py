"""Tests for $image2embedding external."""

from __future__ import annotations

import base64
import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.image2embedding.embedding_format import (
    EMBED_DIM,
    SOURCE_DIM,
    emulated_siglip_embedding,
    pack_siglip_embedding,
    unpack_siglip_embedding,
)
from externals.image2embedding.model_list import get_image2embedding_model
from externals.image2embedding.model_paths import model_dir, model_ready
from externals.image2embedding.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (255, 255, 255)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 "
            b"\x08\x02\x00\x00\x00\xfc\x18\xed\xa3\x00\x00\x00\x0cIDATx\x9cc``\x00"
            b"\x00\x00\x04\x00\x01\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class TestImage2EmbeddingFormat(unittest.TestCase):
    def test_roundtrip(self) -> None:
        import numpy as np

        vec = np.random.default_rng(1).standard_normal(SOURCE_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        encoded = pack_siglip_embedding(vec)
        back = unpack_siglip_embedding(encoded)
        self.assertEqual(len(back), EMBED_DIM)
        self.assertEqual(len(base64.b64decode(encoded)), EMBED_DIM * 2)

    def test_emulated_is_valid(self) -> None:
        encoded = emulated_siglip_embedding("sample.png")
        self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)

class TestImage2EmbeddingParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$image2embedding(model='224', gpu=True)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "image2embedding")
        self.assertEqual(expr.args.get("model"), "224")
        self.assertEqual(expr.args.get("gpu"), "True")


class TestImage2EmbeddingRun(unittest.TestCase):
    def test_emulate_one_embedding_per_image(self) -> None:
        os.environ["AH_EMULATE_IMAGE2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("image2embedding")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            bundle.embeddings.append([0.1, 0.2])
            inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            self.assertEqual(len(out.embeddings), 1)
            encoded = out.embeddings[0]
            self.assertIsInstance(encoded, str)
            self.assertEqual(len(unpack_siglip_embedding(encoded)), EMBED_DIM)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2EMBEDDING", None)

    def test_no_images_clears_embeddings(self) -> None:
        os.environ["AH_EMULATE_IMAGE2EMBEDDING"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("image2embedding")
            )
            inp = ExternalInput(
                bundle=ArrayBundle(embeddings=["deadbeef"]),
                args={},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(out.embeddings, [])
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2EMBEDDING", None)

    def test_resolve_use_gpu_arg(self) -> None:
        from externals.image2embedding.run import _resolve_use_gpu

        inp = ExternalInput(bundle=ArrayBundle(), args={"gpu": "False"}, prompt_text="")
        self.assertFalse(_resolve_use_gpu(inp))


class TestImage2EmbeddingModelPaths(unittest.TestCase):
    def test_model_dir_under_siglip2(self) -> None:
        path = model_dir(get_image2embedding_model("default"))
        self.assertIn("siglip2", str(path).replace("\\", "/"))
        self.assertIn("google-siglip2-base-patch16-384", path.name)

    def test_model_ready_false_when_missing(self) -> None:
        if model_ready():
            self.skipTest("model already downloaded")
        self.assertFalse(model_ready())


if __name__ == "__main__":
    unittest.main()
