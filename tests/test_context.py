"""Tests for %context / track% / %context% storage in action pipelines."""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from ahlib.ah_actions import (
    ContextAction,
    ExternalAction,
    RefAction,
    SequenceAction,
    _parse_context_token,
    _tokenize_actions,
    parse_actions,
)
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir

from tests.test_prompt_merge import _run


def _read_prompts(session_dir: Path, links: list[str]) -> list[str]:
    return [
        (session_dir / link).read_text(encoding="utf-8").strip()
        for link in links
    ]


def _runtime(source: str) -> tuple[Runtime, Path]:
    program = parse_ah_source(source)
    session_dir = create_session_dir(Path("sessions"))
    return Runtime(program, Session(session_dir)), session_dir


class TestContextParse(unittest.TestCase):
    def test_parse_store_session(self) -> None:
        step = parse_actions("%track")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.name, "track")
        self.assertEqual(step.mode, "store")
        self.assertEqual(step.scope, "session")

    def test_parse_load_session(self) -> None:
        step = parse_actions("track%")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.mode, "load")
        self.assertEqual(step.scope, "session")

    def test_parse_store_load_session(self) -> None:
        step = parse_actions("%track%")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.mode, "store_load")
        self.assertEqual(step.scope, "session")

    def test_parse_instruction_scope(self) -> None:
        step = parse_actions("%%track")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.scope, "instruction")
        self.assertEqual(step.mode, "store")

        step = parse_actions("track%%")
        self.assertEqual(step.scope, "instruction")
        self.assertEqual(step.mode, "load")

        step = parse_actions("%%track%%")
        self.assertEqual(step.scope, "instruction")
        self.assertEqual(step.mode, "store_load")

    def test_parse_context_in_sequence(self) -> None:
        expr = parse_actions("@aaa -> %track -> track%")
        self.assertIsInstance(expr, SequenceAction)
        self.assertEqual(len(expr.steps), 2)
        self.assertIsInstance(expr.steps[0], RefAction)
        tail = expr.steps[1]
        self.assertIsInstance(tail, SequenceAction)
        self.assertEqual(len(tail.steps), 2)
        self.assertIsInstance(tail.steps[0], ContextAction)
        self.assertEqual(tail.steps[0].mode, "store")
        self.assertIsInstance(tail.steps[1], ContextAction)
        self.assertEqual(tail.steps[1].mode, "load")

    def test_tokenize_context_with_refs_and_externals(self) -> None:
        tokens = _tokenize_actions(
            "@aaa -> %track -> $llm -> track% -> %ctx%"
        )
        self.assertEqual(
            tokens,
            ["@aaa", "->", "%track", "->", "$llm", "->", "track%", "->", "%ctx%"],
        )

    def test_parse_context_name_with_underscore(self) -> None:
        step = parse_actions("%my_ctx")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.name, "my_ctx")

    def test_parse_context_name_with_digits(self) -> None:
        step = parse_actions("ctx2%")
        self.assertIsInstance(step, ContextAction)
        self.assertEqual(step.name, "ctx2")

    def test_invalid_mismatched_prefix_suffix_raises(self) -> None:
        for tok in ("%track%%", "%%track%", "%%%track", "track%%%" ):
            with self.subTest(tok=tok):
                with self.assertRaises(ValueError):
                    _parse_context_token(tok)

    def test_invalid_token_three_prefix_signs_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_context_token("%%%track")

    def test_tokenizer_rejects_triple_prefix(self) -> None:
        with self.assertRaises(ValueError):
            _tokenize_actions("%%%track")

    def test_tokenizer_rejects_mismatched_scope_markers(self) -> None:
        with self.assertRaises(ValueError):
            _tokenize_actions("%track%%")

    def test_external_tokenization_unaffected(self) -> None:
        tokens = _tokenize_actions("$image(model='x')[3]")
        self.assertEqual(tokens, ["$image(model='x')[3]"])
        expr = parse_actions("$image(model='x')[3]")
        self.assertIsInstance(expr, ExternalAction)
        self.assertEqual(expr.repeat, 3)


class TestContextStoreAndLoad(unittest.TestCase):
    SOURCE = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb -> %track -> track%
"""

    def test_store_passes_input_through(self) -> None:
        source = """
@aaa
aaa

@t: @aaa -> %track
"""
        result, session_dir = _run(source, "t")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa"])

    def test_second_store_passes_current_input_not_stored(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb -> %track
"""
        result, session_dir = _run(source, "run")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa\nbbb"])

    def test_load_returns_stored_session_context(self) -> None:
        result, session_dir = _run(self.SOURCE, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "aaa\nbbb"]))

    def test_store_load_in_one_step(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb -> %track%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "aaa\nbbb"]))

    def test_load_ignores_current_input(self) -> None:
        source = """
@aaa
aaa

@ccc
ccc

@run: @aaa -> %track -> @ccc -> track%
"""
        result, session_dir = _run(source, "run")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa"])

    def test_context_independent_of_same_named_instruction(self) -> None:
        """%data / data% are session context; @data is an instruction — no collision."""
        source = """
@seed
seed-prompt

@data: @seed -> %data

@read: data% -> $pass

@run: @data -> @read
"""
        os.environ["AH_EXTERNAL_INPROCESS"] = "pass"
        try:
            result, session_dir = _run(source, "run")
            self.assertEqual(
                _read_prompts(session_dir, result.prompts), ["seed-prompt"]
            )
        finally:
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)

    def test_self_ref_instruction_raises_recursion_error(self) -> None:
        source = """
@data: @data
body
"""
        runtime, _ = _runtime(source)
        with self.assertRaises(RecursionError) as ctx:
            runtime.run("data")
        msg = str(ctx.exception)
        self.assertIn("@data", msg)
        self.assertIn("data%", msg)

    def test_load_empty_warns(self) -> None:
        source = """
@load: track%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, _ = _run(source, "load")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_load_after_empty_store_warns(self) -> None:
        source = """
@run: %track -> track%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, _ = _run(source, "run")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_store_load_on_empty_input_warns(self) -> None:
        source = """
@run: %track%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, _ = _run(source, "run")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_data_after_empty_store_not_in_context_until_stored(self) -> None:
        source = """
@aaa
aaa

@run: %track -> @aaa -> track%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, session_dir = _run(source, "run")
        self.assertEqual(_read_prompts(session_dir, result.prompts), [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_load_instruction_scope_empty_warns(self) -> None:
        source = """
@load: track%%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, _ = _run(source, "load")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_first_store_with_empty_input_leaves_empty_context(self) -> None:
        source = """
@run: %track
"""
        runtime, _ = _runtime(source)
        runtime.run("run")
        self.assertTrue(runtime._session_contexts["track"].prompts == [])

    def test_store_does_not_modify_downstream_when_only_storing(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb
"""
        runtime, session_dir = _runtime(source)
        result = runtime.run("run")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa\nbbb"])
        self.assertEqual(
            _read_prompts(session_dir, runtime._session_contexts["track"].prompts),
            ["aaa"],
        )

    def test_triple_accumulate_in_session(self) -> None:
        source = """
@a
a

@b
b

@c
c

@run: @a -> %buf -> @b -> %buf -> @c -> %buf -> buf%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(len(texts), 3)
        self.assertIn("a", texts)
        self.assertIn("a\nb", texts)
        self.assertIn("a\nb\nc", texts)


class TestContextScopes(unittest.TestCase):
    def test_instruction_scope_not_shared_across_instructions(self) -> None:
        source = """
@aaa
aaa

@store: @aaa -> %%track
@load: track%%
"""
        runtime, _ = _runtime(source)
        runtime.run("store")
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = runtime.run("load")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_instruction_scope_shared_within_one_instruction(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %%track -> @bbb -> %%track -> track%%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "aaa\nbbb"]))

    def test_session_scope_shared_across_instructions(self) -> None:
        source = """
@aaa
aaa

@store: @aaa -> %track
@load: track%
"""
        runtime, session_dir = _runtime(source)
        runtime.run("store")
        result = runtime.run("load")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa"])

    def test_session_and_instruction_scopes_are_independent(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb -> %%track -> track%%
"""
        runtime, session_dir = _runtime(source)
        result = runtime.run("run")
        session_texts = _read_prompts(
            session_dir,
            runtime._session_contexts["track"].prompts,
        )
        self.assertEqual(session_texts, ["aaa"])
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa\nbbb"])

    def test_nested_instruction_does_not_inherit_instruction_scope(self) -> None:
        source = """
@aaa
aaa

@inner: @aaa -> %%track
@outer: @inner -> track%%
"""
        buf = io.StringIO()
        with redirect_stderr(buf):
            result, _ = _run(source, "outer")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())

    def test_instruction_scope_cleared_on_next_instruction(self) -> None:
        source = """
@aaa
aaa

@save: @aaa -> %%track
@peek: track%%
"""
        runtime, session_dir = _runtime(source)
        runtime.run("save")
        buf = io.StringIO()
        with redirect_stderr(buf):
            result = runtime.run("peek")
        self.assertEqual(result.prompts, [])
        self.assertIn("warning: context 'track' is empty", buf.getvalue())
        self.assertEqual(_read_prompts(session_dir, result.prompts), [])


class TestContextNamesAndIsolation(unittest.TestCase):
    def test_context_independent_from_instruction_name(self) -> None:
        source = """
@track
from_instruction

@ctx: @track -> %track -> track%
"""
        result, session_dir = _run(source, "ctx")
        texts = _read_prompts(session_dir, result.prompts)
        self.assertEqual(texts, ["from_instruction"])

    def test_multiple_context_names_are_independent(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %a -> @bbb -> %b
"""
        runtime, session_dir = _runtime(source)
        runtime.run("run")
        self.assertEqual(
            _read_prompts(session_dir, runtime._session_contexts["a"].prompts),
            ["aaa"],
        )
        self.assertEqual(
            _read_prompts(session_dir, runtime._session_contexts["b"].prompts),
            ["aaa\nbbb"],
        )

    def test_cached_instruction_does_not_repeat_context_store(self) -> None:
        """Second @save with same empty input hits cache; %track runs only once."""
        source = """
@aaa
aaa

@save: @aaa -> %track
@use: track%
"""
        runtime, session_dir = _runtime(source)
        runtime.run("save")
        runtime.run("save")
        result = runtime.run("use")
        self.assertEqual(_read_prompts(session_dir, result.prompts), ["aaa"])
        self.assertEqual(
            _read_prompts(session_dir, runtime._session_contexts["track"].prompts),
            ["aaa"],
        )

    def test_context_store_runs_again_when_input_differs(self) -> None:
        source = """
@aaa
aaa

@save: @aaa -> %track
@run: @save -> @save -> track%
"""
        runtime, session_dir = _runtime(source)
        runtime.run("run")
        texts = sorted(
            _read_prompts(session_dir, runtime._session_contexts["track"].prompts)
        )
        self.assertEqual(texts, sorted(["aaa", "aaa\naaa"]))


class TestContextParallel(unittest.TestCase):
    def test_parallel_then_store(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: (@aaa, @bbb) -> %track -> track%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "bbb"]))

    def test_parallel_then_store_load(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: (@aaa, @bbb) -> %track%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "bbb"]))

    def test_parallel_then_ref_then_store(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@tail
tail

@run: (@aaa, @bbb) -> @tail -> %track -> track%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(len(texts), 2)
        self.assertTrue(all("tail" in t for t in texts))
        self.assertTrue(any(t.startswith("aaa") for t in texts))
        self.assertTrue(any(t.startswith("bbb") for t in texts))


class TestContextFullBundle(unittest.TestCase):
    def test_stores_and_loads_all_array_types(self) -> None:
        runtime, _ = _runtime("@noop:\n")
        for rel in ("p1.txt", "t1.txt", "t2.txt", "i1.png", "s1.mp3", "v1.mp4", "f1.dat"):
            p = runtime.session.base_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".png"):
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
            elif rel.endswith(".mp3"):
                p.write_bytes(b"ID3\x00")
            elif rel.endswith(".mp4"):
                p.write_bytes(b"\x00\x00\x00\x18ftypisom")
            else:
                p.write_bytes(b"x")
        bundle = ArrayBundle(
            prompts=["p1.txt"],
            texts=["t1.txt"],
            images=["i1.png"],
            sounds=["s1.mp3"],
            videos=["v1.mp4"],
            files=["f1.dat"],
            embeddings=[[0.1, 0.2]],
            labels=[["demo", [("images", "i1.png")]]],
        )
        runtime._eval_context_action(
            ContextAction(name="all", mode="store", scope="session"),
            bundle,
            {},
        )
        runtime._eval_context_action(
            ContextAction(name="all", mode="store", scope="session"),
            ArrayBundle(texts=["t2.txt"]),
            {},
        )
        loaded = runtime._eval_context_action(
            ContextAction(name="all", mode="load", scope="session"),
            ArrayBundle(),
            {},
        )
        self.assertEqual(loaded.prompts, ["p1.txt"])
        self.assertEqual(loaded.texts, ["t1.txt", "t2.txt"])
        self.assertEqual(loaded.images, ["i1.png"])
        self.assertEqual(loaded.sounds, ["s1.mp3"])
        self.assertEqual(loaded.videos, ["v1.mp4"])
        self.assertEqual(loaded.files, ["f1.dat"])
        self.assertEqual(loaded.embeddings, [[0.1, 0.2]])
        self.assertEqual(loaded.labels, [["demo", [("images", "i1.png")]]])

    def test_store_load_preserves_distinct_arrays_in_pipeline(self) -> None:
        os.environ["AH_EMULATE_MUSIC"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "music,prompts_to_texts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        try:
            source = """
@caption
irish traditional song

@lyrics
[Verse]
Rain on the window

@lyrics_text: @lyrics -> $prompts_to_texts

@run: (@caption, @lyrics_text) -> %bundle -> bundle%
"""
            runtime, session_dir = _runtime(source)
            result = runtime.run("run")
            stored = runtime._session_contexts["bundle"]
            self.assertTrue(stored.prompts)
            self.assertTrue(stored.texts)
            self.assertTrue(result.prompts)
            self.assertTrue(result.texts)
            self.assertEqual(
                _read_prompts(session_dir, result.prompts),
                _read_prompts(session_dir, stored.prompts),
            )
            self.assertEqual(result.texts, stored.texts)
        finally:
            os.environ.pop("AH_EMULATE_MUSIC", None)
            os.environ.pop("AH_EXTERNAL_INPROCESS", None)
            os.environ.pop("AH_EXTERNAL_SUBPROCESS", None)


class TestContextLoadViaRefConsolidation(unittest.TestCase):
    """After track%, a trailing @ref keeps separate prompt links when N > 1."""

    def test_load_via_ref_keeps_separate_prompts(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@show: track%
@run: @aaa -> %track -> @bbb -> %track -> @show
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "aaa\nbbb"]))

    def test_load_inline_does_not_consolidate(self) -> None:
        source = """
@aaa
aaa

@bbb
bbb

@run: @aaa -> %track -> @bbb -> %track -> track%
"""
        result, session_dir = _run(source, "run")
        texts = sorted(_read_prompts(session_dir, result.prompts))
        self.assertEqual(texts, sorted(["aaa", "aaa\nbbb"]))


if __name__ == "__main__":
    unittest.main()
