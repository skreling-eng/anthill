"""Tests for $face and $face_enhancer (emulated)."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.face.run import run as face_run
from externals.face_enhancer.run import run as face_enhancer_run
from externals.face_lib.model_paths import (
    ALIGN_FILES_2D,
    FACE_ENHANCER_NPY,
    align_ready,
    enhancer_ready,
    models_ready,
)


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    r, g, b = rgb
    raw = b"".join(
        b"\x00" + bytes([r, g, b]) * width for _ in range(height)
    )
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


class TestFaceExternals(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_FACE"] = "1"
        os.environ["AH_EMULATE_FACE_ENHANCER"] = "1"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def tearDown(self) -> None:
        for key in (
            "AH_EMULATE_FACE",
            "AH_EMULATE_FACE_ENHANCER",
            "AH_EXTERNAL_SUBPROCESS",
        ):
            os.environ.pop(key, None)

    def test_face_emulate_replaces_images(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("face")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        link = session.new_link(op_dir, "images", ".png", _png_bytes(64, 48))

        out = face_run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(images=[link]),
                args={"size": "32"},
                prompt_text="",
            ),
        )
        self.assertEqual(len(out.images), 1)
        out_path = session.resolve_link_path(out.images[0])
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 8)

    def test_face_enhancer_emulate(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("face_enhancer")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        link = session.new_link(op_dir, "images", ".png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")
        out = face_enhancer_run(
            ctx,
            ExternalInput(bundle=ArrayBundle(images=[link]), args={}, prompt_text=""),
        )
        self.assertEqual(len(out.images), 1)

    def test_align_ready_checks_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models" / "face"
            root.mkdir(parents=True)
            for name in ALIGN_FILES_2D:
                (root / name).write_bytes(b"x" * 64)
            os.environ["MODELS_PATH"] = str(Path(tmp) / "models")
            try:
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()
                self.assertTrue(align_ready())
                self.assertTrue(models_ready())
            finally:
                os.environ.pop("MODELS_PATH", None)
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()

    def test_enhancer_ready_on_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models" / "face"
            root.mkdir(parents=True)
            (root / FACE_ENHANCER_NPY).write_bytes(b"x" * 1024)
            os.environ["MODELS_PATH"] = str(Path(tmp) / "models")
            try:
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()
                self.assertTrue(enhancer_ready())
            finally:
                os.environ.pop("MODELS_PATH", None)
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()


class TestFaceEnhancerPt(unittest.TestCase):
    def test_load_weights_from_repo_models(self) -> None:
        try:
            import torch
            from externals.face_lib.face_enhancer_pt import FaceEnhancerNet
            from externals.face_lib.model_paths import model_path
        except ImportError:
            self.skipTest("torch not installed")

        weights = model_path(FACE_ENHANCER_NPY)
        if not weights.is_file():
            self.skipTest("FaceEnhancer.npy not present")

        model = FaceEnhancerNet.from_npy(weights)
        model.eval()
        x = torch.zeros(1, 3, 192, 192)
        param = torch.tensor([[0.2]])
        param1 = torch.tensor([[1.0]])
        with torch.inference_mode():
            out = model(x, param, param1)
        self.assertEqual(tuple(out.shape), (1, 3, 768, 768))


if __name__ == "__main__":
    unittest.main()
