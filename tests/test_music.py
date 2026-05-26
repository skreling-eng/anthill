"""Tests for $music caption/lyrics input mapping."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir
from externals.api import ExternalContext, ExternalInput
from externals.music.run import _caption_lyrics_pairs, _read_captions, _read_lyrics, run


class TestMusicInputMapping(unittest.TestCase):
    def test_captions_from_prompts_only(self) -> None:
        session_dir = create_session_dir(Path("sessions"))
        op_dir = session_dir / "1__music"
        op_dir.mkdir(parents=True, exist_ok=True)
        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        cap_link = ctx.new_link("prompts", ".txt", "irish traditional song\n")
        lyr_link = ctx.new_link("texts", ".txt", "[Verse]\nHello world\n")
        inp = ExternalInput(
            bundle=ArrayBundle(prompts=[cap_link], texts=[lyr_link]),
            args={},
            prompt_text="irish traditional song",
        )
        self.assertEqual(_read_captions(ctx, inp), ["irish traditional song"])
        self.assertEqual(_read_lyrics(ctx, inp), ["[Verse]\nHello world"])

    def test_pairs_zip_caption_and_lyrics(self) -> None:
        pairs = _caption_lyrics_pairs(
            ["upbeat dance"],
            ["[Verse]\nRunning through the park tonight"],
        )
        self.assertEqual(
            pairs,
            [("upbeat dance", "[Verse]\nRunning through the park tonight")],
        )

    def test_emulate_uses_texts_as_lyrics_not_prompts(self) -> None:
        os.environ["AH_EMULATE_MUSIC"] = "1"
        session_dir = create_session_dir(Path("sessions"))
        op_dir = session_dir / "1__music"
        op_dir.mkdir(parents=True, exist_ok=True)
        ctx = ExternalContext(session=Session(session_dir), op_dir=op_dir)
        cap_link = ctx.new_link("prompts", ".txt", "style caption\n")
        lyr_link = ctx.new_link("texts", ".txt", "actual lyrics line\n")
        run(
            ctx,
            ExternalInput(
                bundle=ArrayBundle(prompts=[cap_link], texts=[lyr_link]),
                args={"model": "default"},
                prompt_text="style caption",
            ),
        )
        req_files = list(op_dir.rglob("music_request.json"))
        self.assertEqual(req_files, [])


class TestMusicParallelTrack(unittest.TestCase):
    SOURCE = """
@caption
irish traditional song

@lyrics_text: $llm
Write one short lyric line about rain.
Return only the lyric.

@track: (@caption, @lyrics_text) -> $music(model='default')
"""

    def test_parallel_track_single_music_with_both_arrays(self) -> None:
        os.environ["AH_EMULATE_MUSIC"] = "1"
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "music,llm"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(self.SOURCE)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("track")

        music_inputs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in session_dir.rglob("input.json")
            if "__music" in path.parent.name.replace("\\", "/")
        ]
        self.assertEqual(len(music_inputs), 1, music_inputs)
        data = music_inputs[0]
        self.assertTrue(data.get("prompts"), data)
        self.assertTrue(data.get("texts"), data)
        self.assertEqual(len(result.sounds), 1)


if __name__ == "__main__":
    unittest.main()
