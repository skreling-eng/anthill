"""Parallel ( @a, @b ) runs branches then joins contexts before the next -> step."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import ParallelAction, RefAction, SequenceAction, parse_actions
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import Runtime, Session, create_session_dir

from tests.test_prompt_merge import _run


def _read_prompts(session_dir: Path, links: list[str]) -> list[str]:
    return [
        (session_dir / link).read_text(encoding="utf-8").strip()
        for link in links
    ]


class TestParallelJoinBeforeNextStep(unittest.TestCase):
    """Parallel + @ tail runs per branch; parallel + $ joins contexts first."""

    def test_parse_parallel_then_external(self) -> None:
        expr = parse_actions("(@caption, @lyrics) -> $music(model='st')")
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 2)
        self.assertIsInstance(expr.steps[0], ParallelAction)
        self.assertEqual(expr.steps[1].name, "music")

    DDD_SOURCE = """
@aaa
test1

@bbb
test2

@ccc
test3

@ddd: (@aaa, @bbb) -> @ccc
"""

    def test_parse_is_parallel_then_ref(self) -> None:
        actions = parse_ah_source(self.DDD_SOURCE).instructions["ddd"].actions
        expr = parse_actions(actions)
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 2)
        par = expr.steps[0]
        self.assertIsInstance(par, ParallelAction)
        self.assertEqual(
            [b.name for b in par.branches if isinstance(b, RefAction)],
            ["aaa", "bbb"],
        )
        self.assertIsInstance(expr.steps[1], RefAction)
        self.assertEqual(expr.steps[1].name, "ccc")

    def test_ddd_two_composed_prompts_per_branch(self) -> None:
        """Join @aaa+@bbb, run @ccc once; body appended to each joined prompt."""
        result, session_dir = _run(self.DDD_SOURCE, "ddd")
        self.assertEqual(len(result.prompts), 2)
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["test1\ntest3", "test2\ntest3"]))

    def test_ccc_without_input_prompts_creates_one(self) -> None:
        source = """
@ccc
test3

@empty: @ccc
"""
        result, session_dir = _run(source, "empty")
        self.assertEqual(len(result.prompts), 1)
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["test3"])

    def test_ddd_branch_outputs_stay_separate(self) -> None:
        result, session_dir = _run(self.DDD_SOURCE, "ddd")
        texts = _read_prompts(session_dir, result.prompts)
        for text in texts:
            self.assertFalse(
                "test1" in text and "test2" in text,
                f"parallel join must not merge branch outputs: {text!r}",
            )
        self.assertEqual(sum("test1" in t for t in texts), 1)
        self.assertEqual(sum("test2" in t for t in texts), 1)
        self.assertTrue(all("test3" in t for t in texts))

    def test_parallel_without_tail_appends_all_arrays(self) -> None:
        source = """
@aaa
test1

@bbb
test2

@eee: (@aaa, @bbb)
"""
        result, session_dir = _run(source, "eee")
        self.assertEqual(len(result.prompts), 2)
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["test1", "test2"]))

    def test_three_branches_three_outputs(self) -> None:
        source = """
@aaa
a

@bbb
b

@ccc
c

@tail
z

@wide: (@aaa, @bbb, @ccc) -> @tail
"""
        result, session_dir = _run(source, "wide")
        self.assertEqual(len(result.prompts), 3)
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["a\nz", "b\nz", "c\nz"]))

    def test_parallel_refs_fill_different_arrays_then_external_sees_both(self) -> None:
        """(@caption, @lyrics) -> $music: caption in prompts[], lyrics in texts[]."""
        source = """
@caption
irish traditional song

@lyrics
[Verse]
Rain on the window

@lyrics_text: @lyrics -> $prompts_to_texts

@track: (@caption, @lyrics_text) -> $music(model='default')
"""
        os.environ["AH_EMULATE_MUSIC"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "music,prompts_to_texts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("track")

        music_inputs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in session_dir.rglob("input.json")
            if "__music" in path.parent.name.replace("\\", "/")
        ]
        self.assertEqual(len(music_inputs), 1)
        data = music_inputs[0]
        self.assertTrue(data.get("prompts"))
        self.assertTrue(data.get("texts"))
        self.assertEqual(len(result.sounds), 1)

    def test_parallel_mixed_arrays_for_clip(self) -> None:
        """(@images, @audio) -> $image_clip receives joined images[] and sounds[]."""
        source = """
@images: $image[2]
solid red background

@audio: $music(model='default')

@clip: (@images, @audio) -> $image_clip
"""
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EMULATE_MUSIC"] = "1"
        os.environ["AH_EMULATE_IMAGE_CLIP"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "image,music,image_clip"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("clip")

        clip_inputs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in session_dir.rglob("input.json")
            if "__image_clip" in path.parent.name.replace("\\", "/")
        ]
        self.assertEqual(len(clip_inputs), 1)
        data = clip_inputs[0]
        self.assertEqual(len(data.get("images", [])), 2)
        self.assertEqual(len(data.get("sounds", [])), 1)
        self.assertEqual(len(result.videos), 1)

    def test_parallel_externals_without_tail_still_join_outputs(self) -> None:
        """( $ext, $ext ) — two runs on the same bundle, outputs merged."""
        source = """
@prompt
a cat

@pair: @prompt -> ( $image, $image )
"""
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "image"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("pair")
        self.assertEqual(len(result.images), 2)


class TestParallelJoinSequenceSteps(unittest.TestCase):
    def test_parallel_then_external_then_save(self) -> None:
        """Joined context flows through subsequent -> steps in the same chain."""
        source = """
@aaa
line-a

@bbb
line-b

@tail: (@aaa, @bbb) -> $list -> $output
x, y
"""
        os.environ["AH_EXTERNAL_INPROCESS"] = "list,output"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(source)
        session_dir = create_session_dir(Path("sessions"))
        result = Runtime(program, Session(session_dir)).run("tail")
        self.assertTrue(result.files or result.texts)


if __name__ == "__main__":
    unittest.main()
