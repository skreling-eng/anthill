"""Tests for $ocr external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.ocr.langs import resolve_lang, supported_lang_codes
from externals.ocr.model_paths import model_ready, pack_root
from externals.ocr.run import _format_ocr_result, run
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


class TestOcrParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$ocr(lang='en')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "ocr")
        self.assertEqual(expr.args.get("lang"), "en")


class TestOcrFormat(unittest.TestCase):
    def test_format_lines(self) -> None:
        sample = [
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                ("hello", 0.99),
            ],
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                ("world", 0.98),
            ],
        ]
        self.assertEqual(_format_ocr_result([sample]).strip(), "hello\nworld")


class TestOcrRun(unittest.TestCase):
    def test_emulate_one_text_per_image(self) -> None:
        os.environ["AH_EMULATE_OCR"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("ocr")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.images.append(
                ctx.new_link("images", ".png", _png_bytes())
            )
            bundle.texts.append(ctx.new_link("texts", ".txt", "old text\n"))
            inp = ExternalInput(bundle=bundle, args={"lang": "en"}, prompt_text="")
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            self.assertEqual(len(out.texts), 1)
            text = ctx.read_link_text(out.texts[0])
            self.assertIn("[emulated $ocr", text)
        finally:
            os.environ.pop("AH_EMULATE_OCR", None)

    def test_no_images_message(self) -> None:
        os.environ["AH_EMULATE_OCR"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(session=session, op_dir=session.next_op_dir("ocr"))
            inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
            out = run(ctx, inp)
            self.assertIn("no images", ctx.read_link_text(out.texts[0]).lower())
        finally:
            os.environ.pop("AH_EMULATE_OCR", None)


class TestOcrLangs(unittest.TestCase):
    def test_top_languages_resolve(self) -> None:
        es = resolve_lang("es")
        self.assertEqual(es.rec_pack, "latin")
        self.assertEqual(resolve_lang("ru").rec_pack, "cyrillic")
        self.assertEqual(resolve_lang("hi").rec_pack, "devanagari")
        self.assertEqual(resolve_lang("zh").rec_pack, "ch")

    def test_supported_count(self) -> None:
        codes = supported_lang_codes()
        self.assertIn("en", codes)
        self.assertIn("es", codes)
        self.assertIn("ko", codes)
        self.assertGreaterEqual(len(codes), 20)


class TestOcrModelPaths(unittest.TestCase):
    def test_pack_root_under_models(self) -> None:
        root = pack_root("en")
        self.assertIn("ocr", str(root).replace("\\", "/"))
        self.assertEqual(resolve_lang("ja").rec_pack, "japan")


if __name__ == "__main__":
    unittest.main()
