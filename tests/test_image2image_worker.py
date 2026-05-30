"""Tests for $image2image warm worker."""

from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.image2image.worker_client import worker_enabled


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


class TestImage2ImageWorker(unittest.TestCase):
    def test_worker_enabled_by_default(self) -> None:
        os.environ.pop("AH_IMAGE2IMAGE_WORKER", None)
        self.assertTrue(worker_enabled())
        os.environ["AH_IMAGE2IMAGE_WORKER"] = "0"
        self.assertFalse(worker_enabled())
        os.environ.pop("AH_IMAGE2IMAGE_WORKER", None)

    def test_emulate_via_worker(self) -> None:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        os.environ["AH_IMAGE2IMAGE_WORKER"] = "1"
        try:
            from externals.image2image.worker_client import run_via_worker, terminate_worker

            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("image2image")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.images.append(ctx.new_link("images", ".png", _png_bytes()))
            inp = ExternalInput(
                bundle=bundle,
                args={},
                prompt_text="make it blue",
            )
            (op_dir / "input.json").write_text(
                json.dumps(bundle.as_dict(), indent=2),
                encoding="utf-8",
            )
            out = run_via_worker(ctx, inp)
            self.assertEqual(len(out.images), 1)
            terminate_worker()
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)
            os.environ.pop("AH_IMAGE2IMAGE_WORKER", None)


if __name__ == "__main__":
    unittest.main()
