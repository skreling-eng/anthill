"""Build JSONL datasets from Anthill .ah scripts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

_FILENAME_NUMBERED = re.compile(r"^example_(.+)_(\d+)\.ah$", re.IGNORECASE)
_FILENAME_PLAIN = re.compile(r"^example_(.+)\.ah$", re.IGNORECASE)
_REQUEST_RE = re.compile(r"^\s*#\s*request:\s*(.*)\s*$", re.IGNORECASE)

_DEFAULT_EXAMPLE_FOLDERS = ("test_data/examples",)

_SKIP_SUBSTRINGS = ("<prompt>", "User question goes here", "PLACEHOLDER")


def default_example_folders(repo_root: Path | None = None) -> list[Path]:
    root = repo_root or Path(__file__).resolve().parents[2]
    return [(root / rel).resolve() for rel in _DEFAULT_EXAMPLE_FOLDERS]


def parse_request(text: str) -> str:
    for line in text.splitlines():
        m = _REQUEST_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def script_output(text: str) -> str:
    """Target script (drop optional # Request: header line)."""
    lines = text.splitlines()
    if lines and _REQUEST_RE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def _usecase_from_name(name: str) -> tuple[str, int | None]:
    m = _FILENAME_NUMBERED.match(name)
    if m:
        return m.group(1), int(m.group(2))
    m = _FILENAME_PLAIN.match(name)
    if m:
        return m.group(1), None
    return "", None


def _user_prompt(*, request: str, usecase: str, path: Path) -> str:
    if request:
        return f"Write an Anthill (.ah) script for this request:\n{request}"
    if usecase:
        return (
            f"Write an Anthill (.ah) script for the {usecase.replace('_', ' ')} use case "
            f"(pattern: {path.name})."
        )
    return f"Write an Anthill (.ah) script ({path.name})."


def _should_skip(script: str) -> bool:
    if "run @" not in script:
        return True
    low = script.lower()
    return any(s.lower() in low for s in _SKIP_SUBSTRINGS)


def format_user_content(request: str, request_prefix: str = "") -> str:
    """Optional prefix before # Request text (e.g. codegen instruction line)."""
    prefix = request_prefix.strip()
    if not prefix:
        return request
    if prefix.endswith("\n"):
        return prefix + request
    return prefix + "\n" + request


def ah_text_to_row(
    text: str,
    *,
    source: str = "",
    request_prefix: str = "",
) -> dict | None:
    """One training row from inline .ah text (# Request: user, rest assistant)."""
    if _should_skip(text):
        return None
    request = parse_request(text)
    if not request:
        return None
    return {
        "messages": [
            {"role": "user", "content": format_user_content(request, request_prefix)},
            {"role": "assistant", "content": script_output(text)},
        ],
        "source": source,
    }


def rows_from_ah_texts(
    items: list[tuple[str, str]],
    *,
    request_prefix: str = "",
) -> list[dict]:
    """Build rows from (source_label, .ah_text) pairs."""
    rows: list[dict] = []
    for source, text in items:
        row = ah_text_to_row(text, source=source, request_prefix=request_prefix)
        if row is not None:
            rows.append(row)
    return rows


def collect_ah_files(
    folders: list[Path],
    *,
    per_usecase: int | None = None,
) -> list[Path]:
    """Collect .ah files; optional cap per usecase stem for numbered examples."""
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    plain: list[Path] = []

    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".ah":
                continue
            usecase, num = _usecase_from_name(path.name)
            if num is not None:
                grouped[usecase].append((num, path))
            elif path.name.lower().startswith("example_"):
                plain.append(path)

    out: list[Path] = []
    for usecase in sorted(grouped):
        items = sorted(grouped[usecase])[: per_usecase or len(grouped[usecase])]
        out.extend(p for _, p in items)
    out.extend(sorted(plain))
    return out


def example_to_messages(path: Path) -> dict | None:
    raw = path.read_text(encoding="utf-8")
    if _should_skip(raw):
        return None
    request = parse_request(raw)
    usecase, _ = _usecase_from_name(path.name)
    user = _user_prompt(request=request, usecase=usecase, path=path)
    assistant = script_output(raw)
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": path.as_posix(),
    }


def build_dataset(
    folders: list[Path],
    *,
    per_usecase: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for path in collect_ah_files(folders, per_usecase=per_usecase):
        row = example_to_messages(path)
        if row is not None:
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
