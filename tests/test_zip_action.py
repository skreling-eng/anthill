"""Tests for zip(arrays){ body } action."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, SequenceAction, ZipAction, parse_actions
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import Runtime, Session, create_session_dir

from tests.test_prompt_merge import _run


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return b"\x89PNG\r\n\x1a\n"


class TestZipParse(unittest.TestCase):
    def test_parse_zip_block(self) -> None:
        expr = parse_actions("zip(images, texts){ $draw_text }")
        self.assertIsInstance(expr, ZipAction)
        self.assertEqual(expr.array_keys, ["images", "texts"])
        self.assertIsInstance(expr.body, ExternalAction)
        self.assertEqual(expr.body.name, "draw_text")

    def test_parse_zip_in_sequence(self) -> None:
        expr = parse_actions("zip(images, texts){ $draw_text } -> $save('out.png')")
        self.assertIsInstance(expr, SequenceAction)
        self.assertIsInstance(expr.steps[0], ZipAction)


class TestZipRuntime(unittest.TestCase):
    def test_zip_draw_text_per_pair(self) -> None:
        source = """
@img1: $file('test_zip_a.png')
@img2: $file('test_zip_b.png')

@pair: (@img1, @img2)
@cap1
label_a
@cap2
label_b
@run: @pair -> (@cap1, @cap2) -> $prompts_to_texts -> zip(images, texts){ $draw_text }
"""
        repo = Path(__file__).resolve().parents[1]
        (repo / "test_zip_a.png").write_bytes(_png_bytes())
        (repo / "test_zip_b.png").write_bytes(_png_bytes())
        os.environ["AH_EMULATE_DRAW_TEXT"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "file,draw_text,prompts_to_texts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            result, session_dir = _run(source, "run")
            self.assertEqual(len(result.images), 2)
            for link in result.images:
                data = (session_dir / link).read_bytes()
                self.assertIn(b"[emulated $draw_text]", data)
        finally:
            os.environ.pop("AH_EMULATE_DRAW_TEXT", None)
            (repo / "test_zip_a.png").unlink(missing_ok=True)
            (repo / "test_zip_b.png").unlink(missing_ok=True)

    def test_zip_joins_all_output_arrays(self) -> None:
        source = """
@t1
one

@t2
two

@collect: (@t1, @t2) -> $prompts_to_texts

@run: @collect -> zip(texts){ @noop }

@noop:
"""
        os.environ["AH_EXTERNAL_INPROCESS"] = "prompts_to_texts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        result, session_dir = _run(source, "run")
        self.assertEqual(len(result.texts), 2)
        texts = sorted(
            (session_dir / link).read_text(encoding="utf-8").strip()
            for link in result.texts
        )
        self.assertEqual(texts, sorted(["one", "two"]))

    def test_zip_empty_arrays_returns_empty(self) -> None:
        source = """
@run: zip(images, texts){ $draw_text }
"""
        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("run")
        self.assertEqual(result.images, [])
        self.assertEqual(result.texts, [])


if __name__ == "__main__":
    unittest.main()
