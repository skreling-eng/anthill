"""Resolve bundle link paths (session-relative and launch-dir output/ exports)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahlib.ah_runtime import Session


def launch_dir_for(session: Session) -> Path:
    if env := os.environ.get("AH_LAUNCH_DIR"):
        return Path(env).resolve()
    return session.sessions_root.parent.resolve()


def output_export_path(normalized: str, launch: Path) -> Path | None:
    """Map .../output/<session_id>/... links to <launch-dir>/output/<session_id>/..."""
    parts = Path(normalized).parts
    if "output" not in parts:
        return None
    i = parts.index("output")
    if i + 1 >= len(parts):
        return None
    session_id = parts[i + 1]
    rest = Path(*parts[i + 2:]) if i + 2 < len(parts) else Path()
    return (launch / "output" / session_id / rest).resolve()


def normalize_link(link: str) -> str:
    return link.replace("\\", "/").lstrip("./")


def resolve_link_path(
    session: Session,
    link: str,
    *,
    source_session: Path | None = None,
) -> Path:
    path = Path(link)
    if path.is_absolute():
        return path.resolve()

    normalized = normalize_link(link)
    launch = launch_dir_for(session)
    export = output_export_path(normalized, launch)
    if export is not None:
        return export

    candidates: list[Path] = []
    if source_session is not None:
        candidates.append((source_session / link).resolve())
    candidates.extend(
        [
            (session.base_dir / link).resolve(),
            (launch / link).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if source_session is not None:
        return candidates[0]
    return candidates[0]
