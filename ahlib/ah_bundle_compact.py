"""Compact JSON string form of ArrayBundle for debugging and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ahlib.ah_parser import ARRAY_TYPES
from ahlib.ah_runtime import ArrayBundle

_TEXT_ARRAYS = frozenset({"prompts", "texts"})


def _resolve_link_path(base_dir: Path, link: str) -> Path:
    path = Path(link)
    if path.is_absolute():
        return path
    return base_dir / link


def _inline_text_links(base_dir: Path, links: list[str]) -> list[str]:
    values: list[str] = []
    for link in links:
        path = _resolve_link_path(base_dir, link)
        if path.is_file():
            values.append(path.read_text(encoding="utf-8", errors="replace").strip())
        else:
            values.append(link)
    return values


def bundle_compact_dict(
    bundle: ArrayBundle | dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """
    Non-empty arrays only; prompts/texts hold file contents, other arrays keep links.
    Top-level keys are sorted array names.
    """
    if isinstance(bundle, ArrayBundle):
        data = bundle.as_dict()
    else:
        data = bundle

    compact: dict[str, Any] = {}
    for key in sorted(ARRAY_TYPES):
        raw = data.get(key, [])
        if not raw:
            continue
        if key in _TEXT_ARRAYS:
            compact[key] = _inline_text_links(base_dir, list(raw))
        elif key == "changes":
            compact[key] = [list(item) for item in raw]
        else:
            compact[key] = list(raw)
    return compact


def bundle_compact_str(
    bundle: ArrayBundle | dict[str, Any],
    base_dir: Path,
    *,
    indent: bool = False,
) -> str:
    """JSON object string with sorted keys (compact separators unless indent=True)."""
    payload = bundle_compact_dict(bundle, base_dir)
    if indent:
        return json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
