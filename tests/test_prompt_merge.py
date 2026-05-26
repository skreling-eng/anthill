"""Tests for @instruction body + @ref prompt merging before prompt-consuming $."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import action_starts_with_external, parse_actions
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir


def _run(
    source: str,
    target: str,
    *,
    emulate_image: bool = False,
    emulate_llm: bool = False,
    inprocess: str = "image,llm",
) -> tuple[ArrayBundle, Path]:
    if emulate_image:
        os.environ["AH_EMULATE_IMAGE"] = "1"
    else:
        os.environ.pop("AH_EMULATE_IMAGE", None)
    if emulate_llm:
        os.environ["AH_EMULATE_LLM"] = "1"
    else:
        os.environ.pop("AH_EMULATE_LLM", None)
    os.environ["AH_EXTERNAL_INPROCESS"] = inprocess
    os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

    program = parse_ah_source(source)
    session_dir = create_session_dir(Path("sessions"))
    result = Runtime(program, Session(session_dir)).run(target)
    return result, session_dir


def _find_invoke(session_dir: Path, external: str) -> dict:
    matches = sorted(session_dir.rglob("invoke.json"))
    for path in matches:
        if f"__{external}" in path.parent.name.replace("\\", "/"):
            return json.loads(path.read_text(encoding="utf-8"))
    names = [p.parent.name for p in matches]
    raise AssertionError(
        f"No invoke.json for ${external} under {session_dir} (ops: {names})"
    )


def _prompt_text_from_invoke(invoke: dict, session_dir: Path, input_bundle: dict) -> str:
    text = (invoke.get("prompt_text") or "").strip()
    if text:
        return text
    parts: list[str] = []
    for link in input_bundle.get("prompts", []):
        path = Path(link)
        if not path.is_absolute():
            path = session_dir / link
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8").strip())
    return "\n".join(parts)


class TestPromptMergeBeforeExternal(unittest.TestCase):
    """@image: @good_quality -> $image must pass both prompts to $."""

    SOURCE = """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman
"""

    def test_ref_then_image_prompt_concatenated(self) -> None:
        _, session_dir = _run(self.SOURCE, "image", emulate_image=True)
        invoke = _find_invoke(session_dir, "image")
        prompt = invoke["prompt_text"]
        self.assertIn("High Angle Shot", prompt)
        self.assertIn("Best Quality", prompt)
        self.assertIn("running woman", prompt)
        self.assertLess(
            prompt.index("High Angle Shot"),
            prompt.index("running woman"),
            "ref prompt should appear before @image body",
        )

    def test_external_receives_single_joined_prompt_link(self) -> None:
        _, session_dir = _run(self.SOURCE, "image", emulate_image=True)
        for path in session_dir.rglob("input.json"):
            if "__image" not in path.parent.name:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(data.get("prompts", [])), 1)
            link = data["prompts"][0]
            text = (session_dir / link).read_text(encoding="utf-8")
            self.assertIn("High Angle Shot", text)
            self.assertIn("running woman", text)
            self.assertEqual(data.get("changes", []), [])
            return
        self.fail("no $image input.json in session")


class TestExternalFirstBodyPrepend(unittest.TestCase):
    """@x: $llm — instruction body is prepended before the pipeline runs."""

    SOURCE = """
@task: $llm
instruction body for llm
"""

    def test_body_prepended_before_llm(self) -> None:
        _, session_dir = _run(self.SOURCE, "task", emulate_llm=True)
        invoke = _find_invoke(session_dir, "llm")
        self.assertIn("instruction body for llm", invoke["prompt_text"])


class TestParallelRefThenRef(unittest.TestCase):
    """(@aaa, @bbb) -> @ccc composes each branch with @ccc separately."""

    SOURCE = """
@aaa
test1

@bbb
test2

@ccc
test3

@ddd: (@aaa, @bbb) -> @ccc
"""

    def test_ddd_outputs_two_composed_prompts(self) -> None:
        result, session_dir = _run(self.SOURCE, "ddd")
        self.assertEqual(len(result.prompts), 2)
        texts = sorted(
            (session_dir / link).read_text(encoding="utf-8").strip()
            for link in result.prompts
        )
        self.assertEqual(texts, sorted(["test1\ntest3", "test2\ntest3"]))


class TestRefChainBodyMerge(unittest.TestCase):
    """@child: @parent -> @addon — bodies merge when no prompt-consuming $."""

    SOURCE = """
@parent
parent line

@addon
addon line

@child: @parent -> @addon
child line
"""

    def test_ref_chain_composes_all_bodies(self) -> None:
        result, session_dir = _run(self.SOURCE, "child")
        self.assertEqual(len(result.prompts), 1)
        text = (session_dir / result.prompts[0]).read_text(encoding="utf-8")
        self.assertIn("parent line", text)
        self.assertIn("addon line", text)
        self.assertIn("child line", text)


class TestActionStartsWithExternal(unittest.TestCase):
    def test_ref_then_image_not_external_first(self) -> None:
        expr = parse_actions("@good_quality_image -> $image")
        self.assertFalse(action_starts_with_external(expr))

    def test_llm_first_is_external_first(self) -> None:
        expr = parse_actions("$llm -> $texts_to_prompts")
        self.assertTrue(action_starts_with_external(expr))


class TestVideoPromptMultiPromptPreserved(unittest.TestCase):
    """@video_prompt -> $texts_to_prompts must not collapse N prompts before $comfy."""

    SOURCE = """
@video_prompt: $llm[2] -> $texts_to_prompts
shot prompt template

@realistic: @video_prompt -> $comfy(json='wf.json')
"""

    def test_comfy_input_keeps_two_prompt_links(self) -> None:
        os.environ["AH_EMULATE_COMFY"] = "1"
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "llm,comfy"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(self.SOURCE)
        session_dir = create_session_dir(Path("sessions"))
        wf = session_dir / "wf.json"
        wf.write_text(
            '{"9": {"inputs": {"text": "INPUT_PROMPT"}, "class_type": "CLIPTextEncode"}}',
            encoding="utf-8",
        )
        program.instructions["realistic"].actions = (
            program.instructions["realistic"].actions.replace(
                "json='wf.json'", f"json='{wf}'"
            )
        )
        Runtime(program, Session(session_dir)).run("realistic")

        for path in session_dir.rglob("input.json"):
            if "__comfy" not in path.parent.name.replace("\\", "/"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(data.get("prompts", [])),
                2,
                f"expected 2 prompts, got {data.get('prompts')}",
            )
            return
        self.fail("no $comfy input.json")


class TestParallelComfySameInputs(unittest.TestCase):
    """( $comfy, $comfy ) — both branches get the same bundle inputs."""

    SOURCE = """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman

@realistic: ( @image ) -> ( $comfy(port=8000, json='Qwen-Rapid-AIO_4.json'), $comfy(port=8000, json='Qwen-Rapid-AIO_4.json') )
make this image in the realistic style
"""

    def _comfy_invokes(self, session_dir: Path) -> list[tuple[Path, dict, dict]]:
        rows: list[tuple[Path, dict, dict]] = []
        for invoke_path in sorted(session_dir.rglob("invoke.json")):
            if "__comfy" not in invoke_path.parent.name.replace("\\", "/"):
                continue
            invoke = json.loads(invoke_path.read_text(encoding="utf-8"))
            input_path = invoke_path.parent / "input.json"
            inp = json.loads(input_path.read_text(encoding="utf-8"))
            rows.append((invoke_path, invoke, inp))
        return rows

    def test_both_comfy_get_same_prompt_and_image(self) -> None:
        os.environ["AH_EMULATE_COMFY"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "image,comfy"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(self.SOURCE)
        session_dir = create_session_dir(Path("sessions"))
        Runtime(program, Session(session_dir)).run("realistic")

        rows = self._comfy_invokes(session_dir)
        self.assertEqual(len(rows), 2, [str(p) for p, _, _ in rows])

        prompts = []
        images = []
        for _, invoke, inp in rows:
            prompts.append(invoke.get("prompt_text") or "")
            images.append(list(inp.get("images", [])))

        self.assertEqual(prompts[0], prompts[1])
        self.assertIn("make this image in the realistic style", prompts[0])
        self.assertEqual(images[0], images[1])
        self.assertEqual(len(images[0]), 1)


class TestMergePendingInstructionBody(unittest.TestCase):
    def test_merge_applies_join_before_return(self) -> None:
        program = parse_ah_source("@a\nref\n")
        session_dir = create_session_dir(Path("sessions"))
        session = Session(session_dir)
        rt = Runtime(program, session)
        op_dir = session.next_op_dir("test")
        bundle = ArrayBundle()
        bundle.prompts.append(
            session.new_link(op_dir, "prompts", ".txt", "from ref\n")
        )
        pending = ["running woman"]
        merged = rt._merge_pending_instruction_body(bundle, pending, op_dir)
        self.assertFalse(pending)
        self.assertEqual(len(merged.prompts), 1)
        text = (session_dir / merged.prompts[0]).read_text(encoding="utf-8")
        self.assertIn("from ref", text)
        self.assertIn("running woman", text)
        self.assertEqual(merged.changes, [])


if __name__ == "__main__":
    unittest.main()
