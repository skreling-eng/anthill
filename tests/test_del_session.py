"""Tests for $del_session external."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from ahlib.ah_runtime import (
    ArrayBundle,
    Session,
    cleanup_session_after_run,
    create_session_dir,
    run_program,
)
from externals.api import ExternalContext, ExternalInput

run = importlib.import_module("externals.del_session.run").run


class TestDelSessionParse(unittest.TestCase):
    def test_parse(self) -> None:
        expr = parse_actions("$del_session")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "del_session")


class TestDelSessionExternal(unittest.TestCase):
    def test_sets_flag_and_passes_through(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("del_session")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        bundle.texts.append(ctx.new_link("texts", ".txt", "keep\n"))
        inp = ExternalInput(bundle=bundle, args={}, prompt_text="")
        out = run(ctx, inp)
        self.assertTrue(session.delete_after_run)
        self.assertEqual(out.texts, bundle.texts)


class TestDelSessionCleanup(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "pass,del_session"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def tearDown(self) -> None:
        os.environ.pop("AH_EXTERNAL_INPROCESS", None)
        os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)

    def test_removes_session_and_empty_sessions_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_root = root / "sessions"
            source = "@x: $pass -> $del_session\nrun @x\n"
            _meta, session_dir = run_program(
                source,
                sessions_root=sessions_root,
                repo_root=root,
            )
            self.assertFalse(session_dir.exists())
            self.assertFalse(sessions_root.exists())

    def test_keeps_nonempty_sessions_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_root = root / "sessions"
            (sessions_root / "old_run").mkdir(parents=True)
            source = "@x: $pass -> $del_session\nrun @x\n"
            _meta, session_dir = run_program(
                source,
                sessions_root=sessions_root,
                repo_root=root,
            )
            self.assertFalse(session_dir.exists())
            self.assertTrue((sessions_root / "old_run").is_dir())

    def test_no_cleanup_without_del_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_root = root / "sessions"
            source = "@x: $pass\nrun @x\n"
            _meta, session_dir = run_program(
                source,
                sessions_root=sessions_root,
                repo_root=root,
            )
            self.assertTrue(session_dir.is_dir())
            self.assertTrue(sessions_root.is_dir())

    def test_cleanup_helper_removes_created_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_root = root / "sessions"
            session_dir = create_session_dir(sessions_root)
            session = Session(
                session_dir,
                sessions_root=sessions_root,
                sessions_root_created=True,
            )
            session.delete_after_run = True
            cleanup_session_after_run(session)
            self.assertFalse(session_dir.exists())
            self.assertFalse(sessions_root.exists())


if __name__ == "__main__":
    unittest.main()
