"""Tests for $translate external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.translate.model_list import get_translate_model
from externals.translate.model_paths import model_dir, model_ready
from externals.translate.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestTranslateParse(unittest.TestCase):
    def test_parse_scr_dst(self) -> None:
        expr = parse_actions("$translate(scr='ru', dst='en')")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "translate")
        self.assertEqual(expr.args.get("scr"), "ru")
        self.assertEqual(expr.args.get("dst"), "en")


class TestTranslateRun(unittest.TestCase):
    def test_emulate_one_text_per_input(self) -> None:
        os.environ["AH_EMULATE_TRANSLATE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            op_dir = session.next_op_dir("translate")
            ctx = ExternalContext(session=session, op_dir=op_dir)
            bundle = ArrayBundle()
            bundle.texts.append(ctx.new_link("texts", ".txt", "Привет, мир!\n"))
            inp = ExternalInput(
                bundle=bundle,
                args={"scr": "ru", "dst": "en"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertEqual(len(out.texts), 1)
            text = ctx.read_link_text(out.texts[0])
            self.assertIn("[emulated $translate", text)
            self.assertIn("scr=ru", text)
            self.assertIn("dst=en", text)
        finally:
            os.environ.pop("AH_EMULATE_TRANSLATE", None)

    def test_no_texts_message(self) -> None:
        os.environ["AH_EMULATE_TRANSLATE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("translate")
            )
            inp = ExternalInput(
                bundle=ArrayBundle(),
                args={"scr": "ru", "dst": "en"},
                prompt_text="",
            )
            out = run(ctx, inp)
            self.assertIn("no texts", ctx.read_link_text(out.texts[0]).lower())
        finally:
            os.environ.pop("AH_EMULATE_TRANSLATE", None)

    def test_missing_lang_raises(self) -> None:
        os.environ["AH_EMULATE_TRANSLATE"] = "1"
        try:
            session_dir = create_session_dir(Path("sessions"))
            session = Session(session_dir)
            ctx = ExternalContext(
                session=session, op_dir=session.next_op_dir("translate")
            )
            bundle = ArrayBundle()
            bundle.texts.append(ctx.new_link("texts", ".txt", "hello\n"))
            inp = ExternalInput(bundle=bundle, args={"scr": "en"}, prompt_text="")
            with self.assertRaises(ValueError):
                run(ctx, inp)
        finally:
            os.environ.pop("AH_EMULATE_TRANSLATE", None)


class TestTranslateModels(unittest.TestCase):
    def test_default_profile(self) -> None:
        profile = get_translate_model("default")
        self.assertEqual(profile.subdir.as_posix(), "m2m100_1.2B")
        self.assertEqual(profile.hf_repo, "facebook/m2m100_1.2B")

    def test_model_dir_under_models(self) -> None:
        path = model_dir("default")
        self.assertIn("m2m100_1.2B", str(path).replace("\\", "/"))

    def test_model_ready_when_present(self) -> None:
        if model_ready("default"):
            self.assertTrue((model_dir("default") / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
