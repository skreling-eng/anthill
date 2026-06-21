"""Catalog of Anthill $externals from externals/*/_description files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME_RE = re.compile(r"^\$(\w+)")


@dataclass(frozen=True)
class ExternalEntry:
    name: str
    rel_path: str
    summary: str
    description: str


def scan_externals(root: Path) -> list[ExternalEntry]:
    """Read all externals/<name>/_description files under root."""
    externals_dir = root / "externals"
    if not externals_dir.is_dir():
        return []

    entries: list[ExternalEntry] = []
    for desc_path in sorted(externals_dir.glob("*/_description")):
        if not desc_path.is_file():
            continue
        name = desc_path.parent.name
        try:
            text = desc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first_line = text.splitlines()[0].strip() if text.strip() else ""
        match = _NAME_RE.search(first_line)
        if match:
            name = match.group(1)
        summary = first_line or f"${name}"
        entries.append(
            ExternalEntry(
                name=name,
                rel_path=desc_path.relative_to(root).as_posix(),
                summary=summary,
                description=text.strip(),
            )
        )
    entries.sort(key=lambda e: e.name.lower())
    return entries


def format_catalog_markdown(entries: list[ExternalEntry]) -> str:
    if not entries:
        return "No externals found under `externals/*/_description`."
    lines = [f"## Anthill $externals ({len(entries)})\n"]
    for entry in entries:
        lines.append(f"### `${entry.name}`")
        lines.append(f"*{entry.summary}*")
        lines.append(f"\nSource: `{entry.rel_path}`\n")
        preview = entry.description
        if len(preview) > 1200:
            preview = preview[:1200] + "\n\n…"
        lines.append(f"```\n{preview}\n```\n")
    return "\n".join(lines)


def is_externals_catalog_query(request: str) -> bool:
    """Heuristic: user wants a listing/description, not a code change."""
    q = request.strip().lower()
    if not q:
        return False
    change_words = (
        "change",
        "modify",
        "fix",
        "add ",
        "remove",
        "implement",
        "create ",
        "update ",
        "diff",
        "patch",
        "refactor",
        "rewrite",
    )
    if any(w in q for w in change_words):
        return False
    external_words = ("external", "externals", "$")
    list_words = (
        "list",
        "give me",
        "show",
        "what are",
        "enumerate",
        "describe",
        "description",
        "catalog",
        "overview",
        "all ",
    )
    has_external = any(w in q for w in external_words)
    has_list = any(w in q for w in list_words)
    return has_external and has_list
