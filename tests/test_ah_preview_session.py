"""Preview session resolution for nested $ah actions in app.Interface."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from app import Interface, LinkApi
from externals.api import ExternalContext, ExternalInput
from externals.ah.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestAhPreviewSession(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "clear,list,ah"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def test_nested_ah_preview_uses_nested_session_root(self) -> None:
        base = Path(".").resolve()
        api = LinkApi(base)
        ui = Interface(api)
        parent_session = create_session_dir(Path("sessions"))
        ui.session_dir = parent_session

        session = Session(parent_session)
        op_dir = session.next_op_dir("ah")
        ctx = ExternalContext(session=session, op_dir=op_dir, callback=ui)

        script = """
@inner: $list
preview line one, preview line two

run @inner
"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={},
            prompt_text="",
        )
        run(ctx, inp)

        nested_finish = next(
            e for e in ui.data if "FINISH" in e["data"] and "@inner" in e["data"]
        )
        self.assertIn("preview line one", nested_finish["data"])
        self.assertNotIn("[missing:", nested_finish["data"])

        nested_root = (op_dir / "nested").resolve()
        self.assertTrue((nested_root / "1_inner/prompts/0.txt").is_file())
