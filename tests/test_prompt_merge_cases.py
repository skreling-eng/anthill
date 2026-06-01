"""Data-driven prompt-merge tests from PROMPT_MERGE_CASES."""

from __future__ import annotations

import json
import os
import re
import textwrap
import unittest
from pathlib import Path
from typing import Any

from ahlib.ah_bundle_compact import bundle_compact_dict
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_runtime import ArrayBundle, Runtime, Session, create_session_dir

from tests.prompt_merge_cases_data import PROMPT_MERGE_CASES
from tests.test_prompt_merge import _compact, _external_input_compact, _run

_EMULATE_ENV = {
    "image": "AH_EMULATE_IMAGE",
    "llm": "AH_EMULATE_LLM",
    "comfy": "AH_EMULATE_COMFY",
    "image2image": "AH_EMULATE_IMAGE2IMAGE",
    "image2video": "AH_EMULATE_IMAGE2VIDEO",
    "texts2prompts": "AH_EMULATE_LLM",
    "texts_to_prompts": "AH_EMULATE_LLM",
}

_RUN_RE = re.compile(r"^\s*run\s+@(\w+)\s*$", re.MULTILINE | re.IGNORECASE)


def _strip_run_line(script: str) -> str:
    return _RUN_RE.sub("", script).strip() + "\n"


def _parse_run_target(script: str) -> str:
    match = _RUN_RE.search(script)
    if not match:
        raise ValueError(f"script missing 'run @name' line:\n{script}")
    return match.group(1)


def _apply_emulate_externals(names: list[str]) -> str:
    for env_name in _EMULATE_ENV.values():
        os.environ.pop(env_name, None)
    for name in names:
        key = _EMULATE_ENV.get(name)
        if key:
            os.environ[key] = "1"
    os.environ["AH_EXTERNAL_INPROCESS"] = ",".join(names) if names else "image,llm"
    os.environ["AH_EXTERNAL_SUBPROCESS"] = "0"
    return os.environ["AH_EXTERNAL_INPROCESS"]


def _normalize_actual(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if "images" in out:
        out["images"] = ["<image>"] * len(out["images"])
    return out


def _program_source(script: str) -> str:
    return textwrap.dedent(_strip_run_line(script)).strip() + "\n"


def _video_comfy_setup(script: str, run_target: str) -> dict[str, Any]:
    source = _program_source(script)
    _apply_emulate_externals(["llm", "comfy"])
    program = parse_ah_source(source)
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
    Runtime(program, Session(session_dir)).run(run_target)
    comfy_input = next(
        p
        for p in sorted(session_dir.rglob("input.json"))
        if "__comfy" in p.parent.name.replace("\\", "/")
    )
    data = json.loads(comfy_input.read_text(encoding="utf-8"))
    return _normalize_actual(bundle_compact_dict(data, session_dir))


def _parallel_comfy_setup(script: str, run_target: str) -> dict[str, Any]:
    source = _program_source(script)
    _apply_emulate_externals(["image", "comfy"])
    program = parse_ah_source(source)
    session_dir = create_session_dir(Path("sessions"))
    Runtime(program, Session(session_dir)).run(run_target)
    for path in sorted(session_dir.rglob("input.json")):
        if "__comfy" in path.parent.name.replace("\\", "/"):
            data = json.loads(path.read_text(encoding="utf-8"))
            return _normalize_actual(bundle_compact_dict(data, session_dir))
    raise AssertionError("no $comfy input.json")


_CUSTOM_SETUP = {
    "video_comfy": _video_comfy_setup,
    "parallel_comfy": _parallel_comfy_setup,
}


def _compose_actual(case: dict[str, Any]) -> dict[str, Any]:
    program = parse_ah_source("@a\n")
    session_dir = create_session_dir(Path("sessions"))
    session = Session(session_dir)
    rt = Runtime(program, session)
    op_dir = session.next_op_dir("body_rules")
    name = case["name"]
    script = textwrap.dedent(case["script"]).strip()

    if name == "test_body_compose_preserves_other_arrays":
        bundle = ArrayBundle()
        bundle.images.append(session.new_link(op_dir, "images", ".png", b"x"))
        body = "out"
    elif name == "test_rule1_body_concatenated_with_every_input_prompt":
        bundle = ArrayBundle()
        bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "alpha\n"))
        bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "beta\n"))
        body = "suffix"
    elif name == "test_rule2_body_added_when_prompt_links_are_blank":
        bundle = ArrayBundle()
        bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "  \n"))
        body = "only body"
    elif name == "test_rule3_no_body_passes_through_prompts":
        bundle = ArrayBundle()
        bundle.prompts.append(session.new_link(op_dir, "prompts", ".txt", "keep me\n"))
        body = ""
    elif name == "test_rule2_body_added_when_prompts_empty":
        bundle = ArrayBundle()
        body = "only body"
    elif name == "test_rule3_no_body_empty_prompts_stays_empty":
        bundle = ArrayBundle()
        body = ""
    else:
        raise AssertionError(f"unknown compose case: {name}")

    out = rt._compose_instruction_body_into_prompts(bundle, body, op_dir)
    return _normalize_actual(bundle_compact_dict(out, session_dir))


def _run_actual(case: dict[str, Any]) -> dict[str, Any]:
    script = case["script"]
    run_target = _parse_run_target(script)
    source = _program_source(script)
    emulate = list(case.get("emulate_external") or [])
    custom = case.get("custom_setup")

    if custom:
        return _CUSTOM_SETUP[custom](script, run_target)

    inprocess = _apply_emulate_externals(emulate)
    result, session_dir = _run(
        source,
        run_target,
        emulate_image="image" in emulate,
        emulate_llm="llm" in emulate or "texts2prompts" in emulate,
        inprocess=inprocess,
    )
    observe = case.get("observe_external")
    if observe:
        compact = _external_input_compact(session_dir, observe)
    else:
        compact = _compact(result, session_dir)
    return _normalize_actual(json.loads(compact))


def _assert_expected(
    testcase: unittest.TestCase,
    case: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    expected = case["expected"]
    if "prompts" in expected and "prompts" in actual:
        if case["name"] == "test_ddd_outputs_two_composed_prompts":
            testcase.assertEqual(
                sorted(actual["prompts"]),
                sorted(expected["prompts"]),
                case["name"],
            )
            return
        if case["name"] == "test_image_gets_four_separate_prompts_with_suffix":
            testcase.assertEqual(len(actual["prompts"]), 4, case["name"])
            for text in actual["prompts"]:
                testcase.assertIn(
                    "create a beautiful image of the bird in the UK Garden",
                    text,
                )
            testcase.assertEqual(len(set(actual["prompts"])), 4)
            return

    testcase.assertEqual(actual, expected, case["name"])


def _make_case_test(case: dict[str, Any]):
    def test_method(self: TestPromptMergeCases) -> None:
        if case.get("kind") == "compose":
            actual = _compose_actual(case)
        else:
            actual = _run_actual(case)
        _assert_expected(self, case, actual)

    test_method.__doc__ = textwrap.dedent(case["script"]).strip()
    return test_method


class TestPromptMergeCases(unittest.TestCase):
    """One test per PROMPT_MERGE_CASES entry."""


for _case in PROMPT_MERGE_CASES:
    setattr(TestPromptMergeCases, _case["name"], _make_case_test(_case))


if __name__ == "__main__":
    unittest.main()
