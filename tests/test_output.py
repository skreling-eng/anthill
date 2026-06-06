"""Tests for $output external."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput

run = importlib.import_module("externals.output.run").run


class TestOutputParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$output")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "output")


class TestOutputExternal(unittest.TestCase):
    def test_exports_to_launch_dir_not_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch = Path(tmp)
            sessions_root = launch / "sessions"
            session_dir = create_session_dir(sessions_root)
            session = Session(
                session_dir,
                sessions_root=sessions_root,
                sessions_root_created=True,
            )
            op_dir = session.next_op_dir("image")
            image_link = session.new_link(op_dir, "images", ".png", b"\x89PNG\r\n\x1a\n")
            out_op = session.next_op_dir("output")
            ctx = ExternalContext(session=session, op_dir=out_op)
            os.environ["AH_LAUNCH_DIR"] = str(launch)
            try:
                inp = ExternalInput(
                    bundle=ArrayBundle(images=[image_link]),
                    args={},
                    prompt_text="",
                )
                out = run(ctx, inp)
                export_root = launch / "output" / session_dir.name
                self.assertTrue(export_root.is_dir())
                self.assertFalse((sessions_root / "output").exists())
                exported = session.resolve_link_path(out.images[0])
                self.assertEqual(exported.parent.parent.parent, launch / "output")
                self.assertTrue(exported.is_file())
                session.ensure_bundle_files_ready(out)
            finally:
                os.environ.pop("AH_LAUNCH_DIR", None)


if __name__ == "__main__":
    unittest.main()
