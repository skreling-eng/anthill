"""Tests for $image model_names_to_texts option."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.image.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestImageModelNamesToTexts(unittest.TestCase):
    def _run_emulated(
        self,
        *,
        models: list[str] | None = None,
        prompts: list[str] | None = None,
        count: int = 1,
        model_names_to_texts: bool = False,
    ) -> tuple[ArrayBundle, Path]:
        os.environ["AH_EMULATE_IMAGE"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("image")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        bundle = ArrayBundle()
        if prompts:
            for text in prompts:
                bundle.prompts.append(
                    session.new_link(op_dir, "prompts", ".txt", text + "\n")
                )
        args: dict[str, str] = {"count": str(count)}
        arg_lists: dict[str, list[str]] = {}
        if models:
            if len(models) == 1:
                args["model"] = models[0]
            else:
                arg_lists["model"] = models
        if model_names_to_texts:
            args["model_names_to_texts"] = "True"
        inp = ExternalInput(
            bundle=bundle,
            args=args,
            prompt_text="\n".join(prompts or []),
            arg_lists=arg_lists,
            repeat=count if count > 1 and "count" not in args else 1,
        )
        out = run(ctx, inp)
        return out, session_dir

    def tearDown(self) -> None:
        os.environ.pop("AH_EMULATE_IMAGE", None)

    def test_disabled_does_not_add_texts(self) -> None:
        out, _ = self._run_emulated(models=["default"], count=2)
        self.assertEqual(len(out.images), 2)
        self.assertEqual(out.texts, [])

    def test_one_model_five_images(self) -> None:
        out, session_dir = self._run_emulated(
            models=["crazy_desire"],
            count=5,
            model_names_to_texts=True,
        )
        self.assertEqual(len(out.images), 5)
        self.assertEqual(len(out.texts), 5)
        names = [
            (session_dir / link).read_text(encoding="utf-8").strip()
            for link in out.texts
        ]
        self.assertEqual(names, ["crazy_desire"] * 5)

    def test_two_models_two_prompts(self) -> None:
        out, session_dir = self._run_emulated(
            models=["default", "crazy_desire"],
            prompts=["a", "b"],
            count=1,
            model_names_to_texts=True,
        )
        self.assertEqual(len(out.images), 4)
        self.assertEqual(len(out.texts), 4)
        names = [
            (session_dir / link).read_text(encoding="utf-8").strip()
            for link in out.texts
        ]
        self.assertEqual(names, ["default", "default", "crazy_desire", "crazy_desire"])


if __name__ == "__main__":
    unittest.main()
