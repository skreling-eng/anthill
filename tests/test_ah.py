"""Tests for $ah external."""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.ah.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


@dataclass
class RecordingCallback:
    starts: list[str] = field(default_factory=list)
    finishes: list[tuple[str, dict]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def action_start(self, action_name: str) -> None:
        self.starts.append(action_name)

    def action_finish(
        self,
        action_name: str,
        output_context: dict,
        output_json_path: str | None = None,
        session_base_dir: str | None = None,
    ) -> None:
        self.finishes.append(
            (action_name, output_context, output_json_path, session_base_dir)
        )

    def action_error(self, action_name: str, error_message: str) -> None:
        self.errors.append((action_name, error_message))


class TestAhExternal(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AH_EXTERNAL_INPROCESS"] = "clear,list,ah"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    def _ctx(self, callback: RecordingCallback | None = None) -> tuple[Session, ExternalContext]:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("ah")
        ctx = ExternalContext(
            session=session,
            op_dir=op_dir,
            callback=callback,
        )
        return session, ctx

    def test_requires_texts(self) -> None:
        _, ctx = self._ctx()
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(ValueError):
            run(ctx, inp)

    def test_runs_nested_script(self) -> None:
        session, ctx = self._ctx()
        script = """
@hello: $clear
run @hello
"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(out.prompts, [])
        self.assertTrue((ctx.op_dir / "script.ah").is_file())
        self.assertTrue((ctx.op_dir / "nested_run.json").is_file())

    def test_uses_parent_callback(self) -> None:
        callback = RecordingCallback()
        _, ctx = self._ctx(callback)
        script = """
@inner: $list
prompt one, prompt two

run @inner
"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={},
            prompt_text="",
        )
        run(ctx, inp)
        self.assertIn("@inner", callback.starts)
        self.assertIn("$list", callback.starts)
        self.assertIn("@inner", [name for name, *_ in callback.finishes])

    def test_nested_session_under_op_dir(self) -> None:
        session, ctx = self._ctx()
        script = """
@inner: $list
nested line

run @inner
"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={},
            prompt_text="",
        )
        run(ctx, inp)
        nested_root = ctx.op_dir / "nested"
        self.assertTrue(nested_root.is_dir())
        self.assertTrue(any(nested_root.iterdir()))

    def test_entry_override(self) -> None:
        _, ctx = self._ctx()
        script = """
@skip: $clear
@target: $list
one, two

run @skip
"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={"entry": "target"},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.texts), 2)

    def test_strips_markdown_fence(self) -> None:
        _, ctx = self._ctx()
        script = """```ah
@hello: $clear
run @hello
```"""
        link = ctx.new_link("texts", ".txt", script)
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[link]),
            args={},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(out.prompts, [])

    def test_emulate(self) -> None:
        os.environ["AH_EMULATE_AH"] = "1"
        try:
            _, ctx = self._ctx()
            link = ctx.new_link("texts", ".txt", "run @missing\n")
            inp = ExternalInput(
                bundle=ArrayBundle(texts=[link]),
                args={},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            self.assertIn("[emulated $ah", ctx.read_link_text(out.texts[0]))
        finally:
            os.environ.pop("AH_EMULATE_AH", None)


if __name__ == "__main__":
    unittest.main()
