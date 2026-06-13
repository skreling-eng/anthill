"""Tests for $openpose (emulated)."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.openpose.model_paths import BODY_PTH, model_ready, openpose_models_dir
from externals.openpose.run import run as openpose_run


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int] = (120, 80, 40)) -> bytes:
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


class TestOpenposeExternal(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EMULATE_OPENPOSE"] = "1"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def tearDown(self) -> None:
        for key in ("AH_EMULATE_OPENPOSE", "AH_EXTERNAL_SUBPROCESS"):
            os.environ.pop(key, None)

    def test_openpose_emulate_replaces_images(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("openpose")
        ctx = ExternalContext(session=session, op_dir=op_dir)

        link = session.new_link(op_dir, "images", ".png", _png_bytes(96, 64))

        out = openpose_run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(images=[link]),
                args={"detect_resolution": "256"},
                prompt_text="",
            ),
        )
        self.assertEqual(len(out.images), 1)
        out_path = session.resolve_link_path(out.images[0])
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 8)

    def test_model_ready_checks_body_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models" / "openpose"
            root.mkdir(parents=True)
            (root / BODY_PTH).write_bytes(b"x" * 64)
            os.environ["MODELS_PATH"] = str(Path(tmp) / "models")
            try:
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()
                self.assertTrue(model_ready())
                self.assertEqual(openpose_models_dir(), root)
            finally:
                os.environ.pop("MODELS_PATH", None)
                from externals.image.model_paths import models_roots

                models_roots.cache_clear()


if __name__ == "__main__":
    unittest.main()
