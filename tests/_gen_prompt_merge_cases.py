"""Generate tests/prompt_merge_cases_data.py from live runs."""

from __future__ import annotations

import json
import os
import pprint
import textwrap
from pathlib import Path
from typing import Any

from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir

import tests.test_prompt_merge as t


def strip_source(s: str) -> str:
    return textwrap.dedent(s).strip() + "\n"


def normalize_compact(compact: str) -> dict[str, Any]:
    data = json.loads(compact)
    if "images" in data:
        data["images"] = ["<image>"] * len(data["images"])
    return data


def run_case(
    script: str,
    run_target: str,
    *,
    external: str | None = None,
    emulate_image: bool = False,
    emulate_llm: bool = False,
    emulate_comfy: bool = False,
    emulate_image2image: bool = False,
    inprocess: str = "image,llm",
    setup=None,
) -> dict[str, Any]:
    if emulate_image2image:
        os.environ["AH_EMULATE_IMAGE2IMAGE"] = "1"
    else:
        os.environ.pop("AH_EMULATE_IMAGE2IMAGE", None)
    if emulate_comfy:
        os.environ["AH_EMULATE_COMFY"] = "1"
    else:
        os.environ.pop("AH_EMULATE_COMFY", None)

    if setup:
        return setup(script, run_target)

    result, session_dir = t._run(
        script,
        run_target,
        emulate_image=emulate_image,
        emulate_llm=emulate_llm,
        inprocess=inprocess,
    )
    if external:
        compact = t._external_input_compact(session_dir, external)
    else:
        compact = t._compact(result, session_dir)

    return normalize_compact(compact)


def main() -> None:
    cases: list[dict[str, Any]] = []

    script = strip_source(t.TestPromptMergeBeforeExternal.SOURCE)
    expected = run_case(
        t.TestPromptMergeBeforeExternal.SOURCE,
        "image",
        external="image",
        emulate_image=True,
    )
    cases.append(
        {
            "name": "test_ref_then_image_prompt_concatenated",
            "script": script,
            "run_target": "image",
            "external": "image",
            "emulate_image": True,
            "expected": expected,
        }
    )

    script = strip_source(t.TestExternalFirstBodyPrepend.SOURCE)
    cases.append(
        {
            "name": "test_body_prepended_before_llm",
            "script": script,
            "run_target": "task",
            "external": "llm",
            "emulate_llm": True,
            "expected": run_case(
                t.TestExternalFirstBodyPrepend.SOURCE,
                "task",
                external="llm",
                emulate_llm=True,
            ),
        }
    )

    script = strip_source(
        """
        @source: $image
        a portrait photo

        @edit: @source -> $image2image
        make the background blue
        """
    )
    src = script
    cases.append(
        {
            "name": "test_ref_then_image2image_body_on_pipeline_input",
            "script": script,
            "run_target": "edit",
            "external": "image",
            "emulate_image": True,
            "emulate_image2image": True,
            "inprocess": "image,image2image",
            "expected": run_case(
                src,
                "edit",
                external="image",
                emulate_image=True,
                emulate_image2image=True,
                inprocess="image,image2image",
            ),
        }
    )

    script = strip_source(
        """
        @edit: $image2image
        remove extra fingers
        """
    )
    cases.append(
        {
            "name": "test_external_first_body_prepend",
            "script": script,
            "run_target": "edit",
            "external": "image2image",
            "emulate_image2image": True,
            "inprocess": "image2image",
            "expected": run_case(
                script,
                "edit",
                external="image2image",
                emulate_image2image=True,
                inprocess="image2image",
            ),
        }
    )

    script = strip_source(t.TestParallelRefThenRef.SOURCE)
    cases.append(
        {
            "name": "test_ddd_outputs_two_composed_prompts",
            "script": script,
            "run_target": "ddd",
            "expected": run_case(script, "ddd"),
        }
    )

    script = strip_source(t.TestRefChainBodyMerge.SOURCE)
    cases.append(
        {
            "name": "test_ref_chain_composes_all_bodies",
            "script": script,
            "run_target": "child",
            "expected": run_case(script, "child"),
        }
    )

    script = strip_source(t.TestVideoPromptMultiPromptPreserved.SOURCE)

    def video_setup(src: str, target: str) -> dict[str, Any]:
        os.environ["AH_EMULATE_COMFY"] = "1"
        os.environ["AH_EMULATE_LLM"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "llm,comfy"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        program = parse_ah_source(src)
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
        Runtime(program, Session(session_dir)).run(target)
        return normalize_compact(t._external_input_compact(session_dir, "comfy"))

    cases.append(
        {
            "name": "test_comfy_input_keeps_two_prompt_links",
            "script": script,
            "run_target": "realistic",
            "external": "comfy",
            "emulate_comfy": True,
            "emulate_llm": True,
            "inprocess": "llm,comfy",
            "custom_setup": "video_comfy",
            "expected": video_setup(script, "realistic"),
        }
    )

    script = strip_source(t.TestParallelComfySameInputs.SOURCE)

    def parallel_comfy_setup(src: str, target: str) -> dict[str, Any]:
        os.environ["AH_EMULATE_COMFY"] = "1"
        os.environ["AH_EMULATE_IMAGE"] = "1"
        os.environ["AH_EXTERNAL_INPROCESS"] = "image,comfy"
        os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
        program = parse_ah_source(src)
        session_dir = create_session_dir(Path("sessions"))
        Runtime(program, Session(session_dir)).run(target)
        for path in sorted(session_dir.rglob("input.json")):
            if "__comfy" in path.parent.name.replace("\\", "/"):
                data = json.loads(path.read_text(encoding="utf-8"))
                return normalize_compact(t._compact(data, session_dir))
        raise AssertionError("no comfy input")

    cases.append(
        {
            "name": "test_both_comfy_get_same_prompt_and_image",
            "script": script,
            "run_target": "realistic",
            "external": "comfy",
            "emulate_comfy": True,
            "emulate_image": True,
            "inprocess": "image,comfy",
            "custom_setup": "parallel_comfy",
            "expected": parallel_comfy_setup(script, "realistic"),
        }
    )

    script = strip_source(t.TestMultiPromptBodyBeforeImage.SOURCE)
    cases.append(
        {
            "name": "test_image_gets_four_separate_prompts_with_suffix",
            "script": script,
            "run_target": "bird_images",
            "external": "image",
            "emulate_llm": True,
            "emulate_image": True,
            "inprocess": "llm,image,texts2prompts",
            "expected": run_case(
                script,
                "bird_images",
                external="image",
                emulate_llm=True,
                emulate_image=True,
                inprocess="llm,image,texts2prompts",
            ),
        }
    )

    script = strip_source(
        """
        @prompt_only
        hello from body

        @runner: @prompt_only
        """
    )
    cases.append(
        {
            "name": "test_body_only_instruction_writes_output_prompts",
            "script": script,
            "run_target": "runner",
            "expected": run_case(script, "runner"),
        }
    )

    script = strip_source(
        """
        @task: $llm
        instruction for pipeline
        """
    )
    cases.append(
        {
            "name": "test_pipeline_body_is_llm_input",
            "script": script,
            "run_target": "task",
            "external": "llm",
            "emulate_llm": True,
            "expected": run_case(
                script, "task", external="llm", emulate_llm=True
            ),
        }
    )

    program = parse_ah_source("@a\n")
    session_dir = create_session_dir(Path("sessions"))
    session = Session(session_dir)
    rt = Runtime(program, session)
    op_dir = session.next_op_dir("body_rules")

    def compose_case(
        name: str,
        script: str,
        bundle: ArrayBundle,
        body: str,
        expected: dict[str, Any],
    ) -> None:
        out = rt._compose_instruction_body_into_prompts(bundle, body, op_dir)
        cases.append(
            {
                "name": name,
                "script": script,
                "kind": "compose",
                "expected": normalize_compact(t._compact(out, session_dir)),
            }
        )

    compose_case(
        "test_body_compose_preserves_other_arrays",
        strip_source(
            """
            @with_image: $pass
            out
            """
        ),
        (lambda b=ArrayBundle(): (
            b.images.append(session.new_link(op_dir, "images", ".png", b"x")),
            b,
        )[1])(),
        "out",
        {"images": ["<image>"], "prompts": ["out"]},
    )

    bundle = ArrayBundle()
    bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "alpha\n"))
    bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "beta\n"))
    compose_case(
        "test_rule1_body_concatenated_with_every_input_prompt",
        strip_source(
            """
            @two_prompts
            alpha
            beta

            @runner: @two_prompts
            suffix
            """
        ),
        bundle,
        "suffix",
        {"prompts": ["alpha\nsuffix", "beta\nsuffix"]},
    )

    compose_case(
        "test_rule2_body_added_when_prompts_empty",
        strip_source(
            """
            @runner
            only body
            """
        ),
        ArrayBundle(),
        "only body",
        {"prompts": ["only body"]},
    )

    bundle = ArrayBundle()
    bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "  \n"))
    compose_case(
        "test_rule2_body_added_when_prompt_links_are_blank",
        strip_source(
            """
            @blank_prompt
               

            @runner: @blank_prompt
            only body
            """
        ),
        bundle,
        "only body",
        {"prompts": ["only body"]},
    )

    bundle = ArrayBundle()
    bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "keep me\n"))
    compose_case(
        "test_rule3_no_body_passes_through_prompts",
        strip_source(
            """
            @keep
            keep me

            @runner: @keep
            """
        ),
        bundle,
        "",
        {"prompts": ["keep me"]},
    )

    compose_case(
        "test_rule3_no_body_empty_prompts_stays_empty",
        strip_source(
            """
            @runner
            """
        ),
        ArrayBundle(),
        "",
        {},
    )

    out_path = Path(__file__).with_name("prompt_merge_cases_data.py")
    body = pprint.pformat(cases, width=100, sort_dicts=False)
    out_path.write_text(
        "# Generated by tests/_gen_prompt_merge_cases.py\n"
        "from __future__ import annotations\n\n"
        f"PROMPT_MERGE_CASES: list[dict] = {body}\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
