"""Tests for $folder external."""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.folder.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


def _png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (0, 255, 0)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x04\x00\x00\x00\x04"
            b"\x08\x02\x00\x00\x00\x26\x93\x09\x29\x00\x00\x00\x0cIDATx\x9cc``\x00"
            b"\x00\x00\x04\x00\x01\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class TestFolderParse(unittest.TestCase):
    def test_parse_folder_path(self) -> None:
        expr = parse_actions("$folder('test_data/videos')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "folder")
        self.assertEqual(expr.args.get("_path"), "test_data/videos")

    def test_parse_folder_path_and_pattern(self) -> None:
        expr = parse_actions("$folder('src', '*.py')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.args.get("_path"), "src")
        self.assertEqual(expr.args.get("_path2"), "*.py")

    def test_parse_folder_source_path(self) -> None:
        expr = parse_actions("$folder('ahlib', source_path=True)")
        self.assertEqual(expr.args.get("_path"), "ahlib")
        self.assertEqual(expr.args.get("source_path"), "True")


class TestFolderExternal(unittest.TestCase):
    def test_loads_files_by_extension(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        data_dir = repo / "test_data_folder_tmp"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "a.mp4").write_bytes(b"fake-mp4-a")
        (data_dir / "b.mp4").write_bytes(b"fake-mp4-b")
        (data_dir / "c.png").write_bytes(_png_bytes())
        (data_dir / "nested").mkdir(exist_ok=True)
        (data_dir / "nested" / "skip.mp4").write_bytes(b"skip")
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("folder")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": str(data_dir.relative_to(repo))},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 2)
            self.assertEqual(len(out.images), 1)
            v0 = (session_dir / out.videos[0]).read_bytes()
            self.assertEqual(v0, b"fake-mp4-a")
        finally:
            for p in sorted(data_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            data_dir.rmdir()

    def test_source_path_links_without_copy(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        data_dir = repo / "test_data_folder_source_tmp"
        data_dir.mkdir(exist_ok=True)
        src_a = data_dir / "a.py"
        src_b = data_dir / "b.py"
        src_a.write_text("alpha", encoding="utf-8")
        src_b.write_text("beta", encoding="utf-8")
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("folder")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={
                    "_path": str(data_dir.relative_to(repo)),
                    "source_path": "True",
                },
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.files), 2)
            links = sorted(out.files)
            self.assertTrue(Path(links[0]).is_absolute())
            self.assertEqual(Path(links[0]).read_text(encoding="utf-8"), "alpha")
            self.assertEqual(Path(links[1]).read_text(encoding="utf-8"), "beta")
            self.assertFalse((op_dir / "files").exists())
            self.assertEqual(ctx.read_link_text(links[0]), "alpha")
        finally:
            for p in sorted(data_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            data_dir.rmdir()

    def test_filters_by_pattern(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        data_dir = repo / "test_data_folder_pattern_tmp"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "a.py").write_text("a", encoding="utf-8")
        (data_dir / "b.py").write_text("b", encoding="utf-8")
        (data_dir / "c.txt").write_text("c", encoding="utf-8")
        try:
            session_dir = create_session_dir(repo / "sessions")
            session = Session(session_dir)
            op_dir = session.next_op_dir("folder")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={
                    "_path": str(data_dir.relative_to(repo)),
                    "_path2": "*.py",
                },
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.files), 2)
            names = sorted(Path(link).name for link in out.files)
            self.assertEqual(names, ["0.py", "1.py"])
        finally:
            for p in sorted(data_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            data_dir.rmdir()

    def test_emulate_filters_by_pattern(self) -> None:
        os.environ["AH_EMULATE_FOLDER"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("folder")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": "test_data/videos", "_path2": "*.mp4"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.sounds), 0)
            self.assertEqual(len(out.images), 0)
            self.assertEqual(len(out.texts), 0)
        finally:
            os.environ.pop("AH_EMULATE_FOLDER", None)

    def test_emulate_writes_placeholders(self) -> None:
        os.environ["AH_EMULATE_FOLDER"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("folder")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"_path": "test_data/videos"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.videos), 1)
            self.assertEqual(len(out.sounds), 1)
            self.assertEqual(len(out.images), 1)
            self.assertEqual(len(out.texts), 1)
            text = (session_dir / out.videos[0]).read_text(encoding="utf-8")
            self.assertIn("[emulated folder:", text)
        finally:
            os.environ.pop("AH_EMULATE_FOLDER", None)

    def test_integration_via_runtime(self) -> None:
        from tests.test_prompt_merge import _run

        os.environ["AH_EMULATE_FOLDER"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "folder"
        source = """
@run: $folder('test_data/videos')
"""
        result, _ = _run(source, "run")
        self.assertEqual(len(result.videos), 1)


if __name__ == "__main__":
    unittest.main()
