#!/usr/bin/env python3
"""Compare $code request JSON sizes: $folder('examples') vs $ah_code_examples."""

from __future__ import annotations

import json
from pathlib import Path

from externals.ah_code_examples.run import _collect_examples
from externals.folder.run import _resolve_dir_path
from externals.llm.context_limit import estimate_tokens

_REPO = Path(__file__).resolve().parents[1]


def _req(*, prompts: list[str], code_context: str = "", files: list | None = None) -> str:
    return json.dumps(
        {"prompts": prompts, "code_context": code_context, "files": files or []},
        ensure_ascii=False,
        indent=2,
    )


def main() -> None:
    examples_dir = _REPO / "examples"
    ah_files = sorted(
        p for p in examples_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ah"
    )
    folder_files = [
        {
            "path": str(p.relative_to(_REPO)).replace("\\", "/"),
            "name": p.name,
            "content": p.read_text(encoding="utf-8"),
        }
        for p in ah_files
    ]

    dir_path = _resolve_dir_path(type("C", (), {"base_dir": _REPO})(), "test_data/examples")
    usecases = _collect_examples(dir_path, per_usecase=1)
    catalog = {
        "folder": str(dir_path).replace("\\", "/"),
        "per_usecase": 1,
        "usecases": usecases,
    }
    catalog_json = json.dumps(catalog, ensure_ascii=False, indent=2)

    lang_desc = (_REPO / "_lang_desc").read_text(encoding="utf-8")
    codegen_md = (_REPO / "AH_CODEGEN_INSTRUCTIONS.md").read_text(encoding="utf-8")

    prompts = ["Create a script...", "Return only the text of the script."]

    # Chat-sized prompts (truncated __ASK__ + constraints from chat.ah)
    chat_prompts = [
        "Create a script in Anthill that answers the user question or fulfills their request.",
        "First analyze the user message (see AH_CODEGEN_INSTRUCTIONS.md section 8.1):",
        "If not (...): use $llm only (example_llm.ah): @answer: $llm, ... then run @answer.",
        "Return only the text of the script (the whole .ah file, not one line).",
        "The last non-comment line MUST be run @name (e.g. run @answer).",
        "Return only the final result without comments, clarifications, or explanations.",
    ]

    scenarios = {
        "folder(examples) only": _req(prompts=prompts, files=folder_files),
        "ah_code_examples(per_usecase=1) only": _req(
            prompts=prompts, code_context=catalog_json
        ),
        "CHAT: folder + AH_CODEGEN": _req(
            prompts=chat_prompts, files=folder_files + [{
                "path": "AH_CODEGEN_INSTRUCTIONS.md",
                "name": "AH_CODEGEN_INSTRUCTIONS.md",
                "content": codegen_md,
            }]
        ),
        "CHAT: ah per1 + AH_CODEGEN": _req(
            prompts=chat_prompts,
            code_context=catalog_json,
            files=[{
                "path": "AH_CODEGEN_INSTRUCTIONS.md",
                "name": "AH_CODEGEN_INSTRUCTIONS.md",
                "content": codegen_md,
            }],
        ),
        "ah per1 + AH_CODEGEN": _req(
            prompts=prompts,
            code_context=catalog_json,
            files=[
                {
                    "path": "AH_CODEGEN_INSTRUCTIONS.md",
                    "name": "AH_CODEGEN_INSTRUCTIONS.md",
                    "content": codegen_md,
                }
            ],
        ),
        "folder + AH_CODEGEN": _req(
            prompts=prompts,
            files=folder_files
            + [
                {
                    "path": "AH_CODEGEN_INSTRUCTIONS.md",
                    "name": "AH_CODEGEN_INSTRUCTIONS.md",
                    "content": codegen_md,
                }
            ],
        ),
        "ah per1 + _lang_desc + AH_CODEGEN (heavy chat)": _req(
            prompts=prompts,
            code_context=catalog_json,
            files=[
                {"path": "_lang_desc", "name": "_lang_desc", "content": lang_desc},
                {
                    "path": "AH_CODEGEN_INSTRUCTIONS.md",
                    "name": "AH_CODEGEN_INSTRUCTIONS.md",
                    "content": codegen_md,
                },
            ],
        ),
    }

    n_usecases = len(usecases)
    n_examples = sum(len(v) for v in usecases.values())

    print("=== Counts ===")
    print(f"examples/*.ah: {len(ah_files)} files")
    print(
        f"test_data/examples catalog (per_usecase=1): "
        f"{n_usecases} usecases, {n_examples} scripts embedded"
    )
    print()
    print("=== Components ===")
    for label, text in [
        ("AH_CODEGEN_INSTRUCTIONS.md", codegen_md),
        ("_lang_desc", lang_desc),
        ("ah_code_examples JSON (per_usecase=1)", catalog_json),
        ("examples/ raw .ah total", "\n".join(f["content"] for f in folder_files)),
    ]:
        print(f"  {label}: {len(text):,} bytes, ~{estimate_tokens(text):,} tok")
    print()
    print("=== Full $code request.json (indent=2) ===")
    print(f"{'Scenario':<42} {'bytes':>12} {'~tokens':>10}")
    for name, payload in scenarios.items():
        print(f"{name:<42} {len(payload):>12,} {estimate_tokens(payload):>10,}")


if __name__ == "__main__":
    main()
