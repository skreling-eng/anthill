"""Tests for $sound2text external."""

from __future__ import annotations

import json
import os
import unittest
import wave
from pathlib import Path

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.sound2text.run import run


def _tiny_wav(path: Path, *, duration_s: float = 0.05) -> None:
    rate = 16000
    n = int(rate * duration_s)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


class TestSound2TextEmulate(unittest.TestCase):
    def test_emulate_writes_texts(self) -> None:
        os.environ["AH_EMULATE_SOUND2TEXT"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        wav = session_dir / "input.wav"
        _tiny_wav(wav)
        link = str(wav.relative_to(session_dir)).replace("\\", "/")
        op_dir = session_dir / "1__sound2text"
        op_dir.mkdir(parents=True, exist_ok=True)
        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        out = run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(sounds=[link]),
                args={"model": "base"},
                prompt_text="",
            ),
        )
        self.assertEqual(len(out.texts), 1)
        text = ctx.read_link_text(out.texts[0])
        self.assertIn("emulated $sound2text", text)
        self.assertIn("input.wav", text)

    def test_parse_instruction(self) -> None:
        prog = parse_ah_source(
            "@x: @audio -> $sound2text(model='turbo', language='en')\n"
        )
        self.assertIn("sound2text", prog.instructions["x"].actions)

    def test_emulate_json_writes_word_timestamps(self) -> None:
        os.environ["AH_EMULATE_SOUND2TEXT"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        wav = session_dir / "input.wav"
        _tiny_wav(wav)
        link = str(wav.relative_to(session_dir)).replace("\\", "/")
        op_dir = session_dir / "1__sound2text"
        op_dir.mkdir(parents=True, exist_ok=True)
        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        out = run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(sounds=[link]),
                args={"model": "base", "json": "True"},
                prompt_text="",
            ),
        )
        self.assertEqual(len(out.texts), 1)
        self.assertTrue(out.texts[0].endswith(".json"))
        data = json.loads(ctx.read_link_text(out.texts[0]))
        self.assertEqual(data["file"], "input.wav")
        self.assertIn("words", data)
        self.assertGreaterEqual(len(data["words"]), 1)
        word = data["words"][0]
        self.assertIn("word", word)
        self.assertIn("start", word)
        self.assertIn("end", word)


if __name__ == "__main__":
    unittest.main()
