"""Tests for @instruction body + @ref prompt merging before prompt-consuming $."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ahlib.ah_actions import action_starts_with_external, parse_actions
from ahlib.ah_bundle_compact import bundle_compact_dict, bundle_compact_str
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


def _compact(bundle: ArrayBundle | dict, session_dir: Path) -> str:
    return bundle_compact_str(bundle, session_dir)


def _compact_dict(bundle: ArrayBundle | dict, session_dir: Path) -> dict:
    return bundle_compact_dict(bundle, session_dir)


def _external_input_compact(session_dir: Path, external: str) -> str:
    for path in sorted(session_dir.rglob("input.json")):
        if f"__{external}" in path.parent.name.replace("\\", "/"):
            data = json.loads(path.read_text(encoding="utf-8"))
            return _compact(data, session_dir)
    ops = [p.parent.name for p in session_dir.rglob("invoke.json")]
    raise AssertionError(
        f"No input.json for ${external} under {session_dir} (ops: {ops})"
    )


class TestPromptMergeBeforeExternal(unittest.TestCase):
    """@image: @good_quality -> $image must pass both prompts to $."""

    SOURCE = """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman
"""

    def test_ref_then_image_prompt_concatenated(self) -> None:
        """Merged prompt text, caller-before-ref order, single prompts[] link, no changes."""
        _, session_dir = _run(self.SOURCE, "image", emulate_image=True)
        compact = _external_input_compact(session_dir, "image")
        data = json.loads(compact)
        self.assertEqual(len(data["prompts"]), 1)
        self.assertNotIn("changes", data)
        self.assertIn("High Angle Shot", compact)
        self.assertIn("Best Quality", compact)
        self.assertIn("running woman", compact)
        self.assertLess(
            data["prompts"][0].index("running woman"),
            data["prompts"][0].index("High Angle Shot"),
            "@image body is on pipeline input before @ref appends its body",
        )


class TestExternalFirstBodyPrepend(unittest.TestCase):
    """@x: $llm — instruction body is prepended before the pipeline runs."""

    SOURCE = """
@task: $llm
instruction body for llm
"""

    def test_body_prepended_before_llm(self) -> None:
        _, session_dir = _run(self.SOURCE, "task", emulate_llm=True)
        compact = _external_input_compact(session_dir, "llm")
        self.assertIn("instruction body for llm", compact)


class TestImage2ImagePromptMerge(unittest.TestCase):
    """@ref -> $image2image and @x: $image2image must pass instruction body as prompt."""

    def test_ref_then_image2image_body_on_pipeline_input(self) -> None:
        """@edit body is pipeline input; @source $image sees it ($image clears prompts[])."""
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        try:
            source = """
@source: $image
a portrait photo

@edit: @source -> $image2image
make the background blue
"""
            _, session_dir = _run(
                source, "edit", inprocess="image,image2image"
            )
            compact = _external_input_compact(session_dir, "image")
            self.assertIn("make the background blue", compact)
            self.assertIn("a portrait photo", compact)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)
            os.environ.pop("AH_EMULATE_IMAGE", None)

    def test_external_first_body_prepend(self) -> None:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
        try:
            source = """
@edit: $image2image
remove extra fingers
"""
            _, session_dir = _run(source, "edit", inprocess="image2image")
            compact = _external_input_compact(session_dir, "image2image")
            self.assertIn("remove extra fingers", compact)
        finally:
            os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)


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
        data = _compact_dict(result, session_dir)
        self.assertEqual(
            sorted(data["prompts"]),
            sorted(["test1\ntest3", "test2\ntest3"]),
        )


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
        compact = _compact(result, session_dir)
        self.assertIn("parent line", compact)
        self.assertIn("addon line", compact)
        self.assertIn("child line", compact)
        data = json.loads(compact)
        self.assertEqual(len(data["prompts"]), 1)


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

        compact = _external_input_compact(session_dir, "comfy")
        data = json.loads(compact)
        self.assertEqual(len(data.get("prompts", [])), 2, compact)


class TestParallelComfySameInputs(unittest.TestCase):
    """( $comfy, $comfy ) — both branches get the same bundle inputs."""

    SOURCE = """
@good_quality_image
High Angle Shot, Best Quality

@image: @good_quality_image -> $image
running woman

@realistic_style
make this image in the realistic style

@realistic: @image -> @realistic_style -> (
  $comfy(port=8000, json='Qwen-Rapid-AIO_4.json'),
  $comfy(port=8000, json='Qwen-Rapid-AIO_4.json')
)
"""

    def test_both_comfy_get_same_prompt_and_image(self) -> None:
        os.environ["AH_EMULATE_COMFY"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "image,comfy"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(self.SOURCE)
        session_dir = create_session_dir(Path("sessions"))
        Runtime(program, Session(session_dir)).run("realistic")

        compacts: list[str] = []
        for path in sorted(session_dir.rglob("input.json")):
            if "__comfy" not in path.parent.name.replace("\\", "/"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            compacts.append(_compact(data, session_dir))

        self.assertEqual(len(compacts), 2, compacts)
        self.assertEqual(compacts[0], compacts[1])
        self.assertIn("make this image in the realistic style", compacts[0])
        data = json.loads(compacts[0])
        self.assertEqual(len(data["images"]), 1)


class TestMultiPromptBodyBeforeImage(unittest.TestCase):
    """@bird_images: @bird_names -> $image — suffix on each prompt, not one join."""

    SOURCE = """
@bird_names: $llm[4] -> $texts2prompts
name a UK garden bird

@bird_images: @bird_names -> $image
create a beautiful image of the bird in the UK Garden
"""

    def test_image_gets_four_separate_prompts_with_suffix(self) -> None:
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "llm,image,texts2prompts"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"

        program = parse_ah_source(self.SOURCE)
        session_dir = create_session_dir(Path("sessions"))
        Runtime(program, Session(session_dir)).run("bird_images")

        compact = _external_input_compact(session_dir, "image")
        data = json.loads(compact)
        prompts = data["prompts"]
        self.assertEqual(len(prompts), 4, compact)
        for text in prompts:
            self.assertIn(
                "create a beautiful image of the bird in the UK Garden",
                text,
            )
        self.assertEqual(len(set(prompts)), 4, "each prompt should differ")


class TestPipelineVsOutputBody(unittest.TestCase):
    """@act: pipeline + body vs @act + body only."""

    def test_body_only_instruction_writes_output_prompts(self) -> None:
        source = """
@prompt_only
hello from body

@runner: @prompt_only
"""
        result, session_dir = _run(source, "runner")
        compact = _compact(result, session_dir)
        self.assertIn("hello from body", compact)
        self.assertEqual(json.loads(compact)["prompts"], ["hello from body"])

    def test_pipeline_body_is_llm_input(self) -> None:
        source = """
@task: $llm
instruction for pipeline
"""
        _, session_dir = _run(source, "task", emulate_llm=True)
        compact = _external_input_compact(session_dir, "llm")
        self.assertIn("instruction for pipeline", compact)


class TestInstructionBodyPromptRules(unittest.TestCase):
    """Unit tests for Runtime._compose_instruction_body_into_prompts."""

    def setUp(self) -> None:
        program = parse_ah_source("@a\n")
        self.session_dir = create_session_dir(Path("sessions"))
        self.session = Session(self.session_dir)
        self.rt = Runtime(program, self.session)
        self.op_dir = self.session.next_op_dir("body_rules")

    def _compose(self, bundle: ArrayBundle, body: str) -> ArrayBundle:
        return self.rt._compose_instruction_body_into_prompts(
            bundle, body, self.op_dir
        )

    def test_body_compose_preserves_other_arrays(self) -> None:
        bundle = ArrayBundle()
        bundle.images.append(
            self.session.new_link(self.op_dir, "images", ".png", b"x")
        )
        out = self._compose(bundle, "out")
        data = _compact_dict(out, self.session_dir)
        self.assertEqual(data["prompts"], ["out"])
        self.assertEqual(len(data["images"]), 1)

    def test_rule1_body_concatenated_with_every_input_prompt(self) -> None:
        bundle = ArrayBundle()
        bundle.prompts.append(
            self.session.new_link(self.op_dir, "prompts", ".txt", "alpha\n")
        )
        bundle.prompts.append(
            self.session.new_link(self.op_dir, "prompts", ".txt", "beta\n")
        )
        out = self._compose(bundle, "suffix")
        data = _compact_dict(out, self.session_dir)
        self.assertEqual(len(data["prompts"]), 2)
        self.assertIn("alpha", data["prompts"][0])
        self.assertIn("suffix", data["prompts"][0])
        self.assertIn("beta", data["prompts"][1])
        self.assertIn("suffix", data["prompts"][1])

    def test_rule2_body_added_when_prompts_empty(self) -> None:
        out = self._compose(ArrayBundle(), "only body")
        self.assertEqual(
            json.loads(_compact(out, self.session_dir)),
            {"prompts": ["only body"]},
        )

    def test_rule2_body_added_when_prompt_links_are_blank(self) -> None:
        bundle = ArrayBundle()
        bundle.prompts.append(
            self.session.new_link(self.op_dir, "prompts", ".txt", "  \n")
        )
        out = self._compose(bundle, "only body")
        self.assertEqual(
            json.loads(_compact(out, self.session_dir)),
            {"prompts": ["only body"]},
        )

    def test_rule3_no_body_passes_through_prompts(self) -> None:
        bundle = ArrayBundle()
        bundle.prompts.append(
            self.session.new_link(self.op_dir, "prompts", ".txt", "keep me\n")
        )
        out = self._compose(bundle, "")
        data = _compact_dict(out, self.session_dir)
        self.assertEqual(data["prompts"], ["keep me"])
        self.assertEqual(out.prompts[0], bundle.prompts[0])

    def test_rule3_no_body_empty_prompts_stays_empty(self) -> None:
        out = self._compose(ArrayBundle(), "")
        self.assertEqual(_compact(out, self.session_dir), "{}")


if __name__ == "__main__":
    unittest.main()
