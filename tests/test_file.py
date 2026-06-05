"""Tests for $file external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.file.run import run
from externals.file_extensions import array_for_extension
from externals.invoke import subprocess_enabled
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08"
            b"\x08\x02\x00\x00\x00=\x91\x10\x20\x00\x00\x00\x0cIDATx\x9cc``\x00\x00"
            b"\x00\x04\x00\x01\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class TestFileExtensions(unittest.TestCase):
    def test_ass_routes_to_texts(self) -> None:
        self.assertEqual(array_for_extension(".ass"), "texts")
        self.assertEqual(array_for_extension(".ASS"), "texts")

    def test_unknown_routes_to_files(self) -> None:
        self.assertEqual(array_for_extension(".xyz"), "files")


class TestFileExternal(unittest.TestCase):
    def test_file_inprocess_by_default(self) -> None:
        for key in ("AH_EXTERNAL_INPROCESS", "AH_EXTERNAL_SUBPROCESS"):
            os.environ.pop(key, None)
        self.assertFalse(subprocess_enabled("file"))
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "file"
        self.assertTrue(subprocess_enabled("file"))
        os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)

    def test_parse_file_source_path(self) -> None:
        expr = parse_actions("$file('_lang_desc', source_path=True)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "file")
        self.assertEqual(expr.args.get("_path"), "_lang_desc")
        self.assertEqual(expr.args.get("source_path"), "True")

    def test_loads_real_image_bytes(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        asset = repo / "test_asset_file.png"
        asset.write_bytes(_png_bytes())
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("file")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": "test_asset_file.png"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.images), 1)
            data = (session_dir / out.images[0]).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG"))
            self.assertNotIn(b"[emulated file:", data)
        finally:
            asset.unlink(missing_ok=True)

    def test_resolves_lang_desc_at_repo_root(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        lang_desc = repo / "_lang_desc"
        if not lang_desc.is_file():
            self.skipTest("_lang_desc missing at repo root")
        session_dir = create_session_dir(repo / "sessions")
        session = Session(session_dir)
        op_dir = session.next_op_dir("file")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(
            bundle=ArrayBundle(),
            args={"_path": "_lang_desc", "source_path": "True"},
            prompt_text="",
        )
        from unittest import mock

        with mock.patch("externals.file.run._REPO_ROOT", repo):
            out = run(ctx, inp)
        self.assertEqual(len(out.files), 1)
        self.assertEqual(Path(out.files[0]).resolve(), lang_desc.resolve())

    def test_loads_ass_into_texts(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        asset = repo / "test_asset_file.ass"
        asset.write_text("[Script Info]\nTitle: test\n", encoding="utf-8")
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("file")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": "test_asset_file.ass"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            data = (session_dir / out.texts[0]).read_text(encoding="utf-8")
            self.assertIn("[Script Info]", data)
        finally:
            asset.unlink(missing_ok=True)

    def test_source_path_links_without_copy(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        asset = repo / "test_asset_file_source.txt"
        asset.write_text("hello source", encoding="utf-8")
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("file")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={
                    "_path": "test_asset_file_source.txt",
                    "source_path": "True",
                },
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            link = out.texts[0]
            self.assertTrue(Path(link).is_absolute())
            self.assertEqual(Path(link).read_text(encoding="utf-8"), "hello source")
            self.assertFalse((op_dir / "texts").exists())
            self.assertEqual(ctx.read_link_text(link), "hello source")
        finally:
            asset.unlink(missing_ok=True)

    def test_emulate_writes_placeholder(self) -> None:
        os.environ["AH_EMULATE_FILE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("file")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": "missing.png"},
                prompt_text="",
            )
            out = run(ctx, inp)
            text = (session_dir / out.images[0]).read_text(encoding="utf-8")
            self.assertIn("[emulated file:", text)
        finally:
            os.environ.pop("AH_EMULATE_FILE", None)


if __name__ == "__main__":
    unittest.main()
