"""Tests for $text2speech external."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from ahlib.ah_actions import ExternalAction, parse_actions
from externals.api import ExternalContext, ExternalInput
from externals.text2speech.espeak import _DEFAULT_DLL
from externals.text2speech.model_paths import (
    DEFAULT_VOICE,
    KOKORO_DIR,
    lang_code_for_voice,
    legacy_available,
    resolve_voice_pack,
    use_legacy_backend,
)
from externals.text2speech.run import run
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir


class TestText2SpeechParse(unittest.TestCase):
    def test_parse_args(self) -> None:
        expr = parse_actions("$text2speech(voice=af_bella, speed=2, device=cpu)")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.name, "text2speech")
        self.assertEqual(expr.args.get("voice"), "af_bella")
        self.assertEqual(expr.args.get("speed"), "2")


class TestText2SpeechModelPaths(unittest.TestCase):
    def test_default_espeak_dll_path(self) -> None:
        self.assertEqual(_DEFAULT_DLL.name, "libespeak-ng.dll")
        self.assertEqual(_DEFAULT_DLL.parent.name, "tools")

    def test_lang_code(self) -> None:
        self.assertEqual(lang_code_for_voice("af_bella"), "a")
        self.assertEqual(lang_code_for_voice("bm_george"), "b")

    def test_resolve_voice_name_local(self) -> None:
        local = KOKORO_DIR / "voices" / "af_bella.pt"
        if not local.is_file():
            self.skipTest("models/kokoro/voices/af_bella.pt not present")
        with mock.patch.dict(os.environ, {"AH_KOKORO_DIR": ""}, clear=False):
            path = resolve_voice_pack("af_bella", KOKORO_DIR, download=False)
        self.assertEqual(path, local.resolve())

    @mock.patch("externals.text2speech.assets.ensure_voice_pack")
    def test_resolve_voice_downloads_to_models(self, ensure) -> None:
        target = KOKORO_DIR / "voices" / "_nonexistent_test_voice.pt"
        ensure.return_value = target
        with mock.patch.dict(os.environ, {"AH_KOKORO_DIR": ""}, clear=False):
            path = resolve_voice_pack("_nonexistent_test_voice", KOKORO_DIR)
        ensure.assert_called_once()
        self.assertEqual(path, target)

    def test_backend_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AH_TEXT2SPEECH_BACKEND": "pipeline"},
            clear=False,
        ):
            self.assertFalse(use_legacy_backend())


class TestText2SpeechRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get("AH_EMULATE_TEXT2SPEECH")
        os.environ["AH_EMULATE_TEXT2SPEECH"] = "1"

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("AH_EMULATE_TEXT2SPEECH", None)
        else:
            os.environ["AH_EMULATE_TEXT2SPEECH"] = self._prev

    def test_emulated_run(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("text2speech")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        text_link = ctx.new_link("texts", ".txt", "Hello world.\n")
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[text_link]),
            args={"voice": DEFAULT_VOICE},
            prompt_text="",
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 1)

    def test_requires_text(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("text2speech")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        inp = ExternalInput(bundle=ArrayBundle(), args={}, prompt_text="")
        with self.assertRaises(ValueError):
            run(ctx, inp)

    def test_multiple_voices_emulated(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        op_dir = session.next_op_dir("text2speech")
        ctx = ExternalContext(session=session, op_dir=op_dir)
        text_link = ctx.new_link("texts", ".txt", "One line.")
        inp = ExternalInput(
            bundle=ArrayBundle(texts=[text_link]),
            args={},
            prompt_text="",
            arg_lists={"voice": ["af_bella", "bm_george"]},
        )
        out = run(ctx, inp)
        self.assertEqual(len(out.sounds), 2)


class TestLegacyDetection(unittest.TestCase):
    def test_legacy_available_false_by_default(self) -> None:
        if not legacy_available():
            self.assertFalse((KOKORO_DIR / "models.py").is_file())
