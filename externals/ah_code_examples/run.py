"""$ah_code_examples — scan example_*.ah files and return grouped JSON in texts[]."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.folder.run import _resolve_dir_path
from ahlib.ah_runtime import ArrayBundle

_FILENAME_RE = re.compile(r"^example_(.+)_(\d+)\.ah$", re.IGNORECASE)
_REQUEST_RE = re.compile(r"^\s*#\s*request:\s*(.*)\s*$", re.IGNORECASE)
_MAX_PER_USECASE = 20

_HELP = """
$ah_code_examples needs a folder path.

Example:
  @refs: $ah_code_examples(folder='test_data/examples', per_usecase=20)

Scans example_<usecase>_<n>.ah files (non-recursive), up to per_usecase examples
per use case (max 20). Writes one JSON text link to texts[].
"""


def _folder_from_args(args: dict[str, str]) -> str:
    return (
        args.get("folder", "")
        or args.get("_path", "")
        or args.get("path", "")
    ).strip()


def _per_usecase_from_args(args: dict[str, str]) -> int:
    raw = args.get("per_usecase", "20").strip() or "20"
    try:
        n = int(raw)
    except ValueError:
        n = 20
    return max(1, min(_MAX_PER_USECASE, n))


def _parse_request(text: str) -> str:
    for line in text.splitlines():
        m = _REQUEST_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def _collect_examples(dir_path: Path, *, per_usecase: int) -> dict[str, list[dict]]:
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(dir_path.iterdir()):
        if not path.is_file():
            continue
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        usecase, num_s = m.group(1), m.group(2)
        grouped[usecase].append((int(num_s), path))

    out: dict[str, list[dict]] = {}
    for usecase in sorted(grouped):
        items: list[dict] = []
        for _num, path in sorted(grouped[usecase])[:per_usecase]:
            script = path.read_text(encoding="utf-8")
            items.append(
                {
                    "file": path.name,
                    "request": _parse_request(script),
                    "script": script,
                }
            )
        out[usecase] = items
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    folder = _folder_from_args(inp.args)
    if not folder:
        raise RuntimeError(_HELP.strip())

    per_usecase = _per_usecase_from_args(inp.args)
    dir_path = _resolve_dir_path(ctx, folder)
    usecases = _collect_examples(dir_path, per_usecase=per_usecase)

    payload = {
        "folder": str(dir_path).replace("\\", "/"),
        "per_usecase": per_usecase,
        "usecases": usecases,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    out = inp.bundle.copy()
    out.texts.clear()
    out.texts.append(ctx.new_link("texts", ".txt", text))
    return out
