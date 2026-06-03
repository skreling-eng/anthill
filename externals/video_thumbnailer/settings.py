"""Argument bundle for $video_thumbnailer preview sheets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from externals.api import ExternalInput

_PREFIX = "$video_thumbnailer"


@dataclass(frozen=True)
class ThumbnailOptions:
    width: int = 1024
    columns: int = 5
    rows: int = 5
    vertical_video_columns: int | None = None
    vertical_video_rows: int | None = None
    spacing: int = 2
    background_color: str = "black"
    no_header: bool = False
    header_font: str | None = None
    header_font_size: int = 14
    header_font_color: str = "white"
    timestamp_font: str | None = None
    timestamp_font_size: int = 12
    timestamp_font_color: str = "white"
    timestamp_shadow_color: str | None = "black"
    comment_label: str = "Comment:"
    comment_text: str | None = None
    skip_seconds: float = 10.0
    jpeg_quality: int = 95


def _bool_arg(inp: ExternalInput, key: str, default: bool) -> bool:
    raw = inp.args.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_arg(
    inp: ExternalInput,
    key: str,
    default: int,
    *,
    min_value: int = 1,
) -> int:
    raw = inp.args.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_PREFIX}: invalid {key}={raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{_PREFIX}: {key} must be >= {min_value}, got {value}")
    return value


def _float_arg(
    inp: ExternalInput,
    key: str,
    default: float,
    *,
    min_value: float = 0.0,
) -> float:
    raw = inp.args.get(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_PREFIX}: invalid {key}={raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{_PREFIX}: {key} must be >= {min_value}, got {value}")
    return value


def _optional_int_arg(inp: ExternalInput, key: str) -> int | None:
    raw = inp.args.get(key, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_PREFIX}: invalid {key}={raw!r}") from exc


def _optional_str_arg(inp: ExternalInput, key: str) -> str | None:
    raw = inp.args.get(key, "").strip()
    return raw or None


def _shadow_color_arg(inp: ExternalInput) -> str | None:
    raw = inp.args.get("timestamp_shadow_color", "black").strip()
    if not raw or raw.lower() in ("none", "off", "0"):
        return None
    return raw


def options_from_input(inp: ExternalInput) -> ThumbnailOptions:
    return ThumbnailOptions(
        width=_int_arg(inp, "width", 1024, min_value=64),
        columns=_int_arg(inp, "columns", 5, min_value=1),
        rows=_int_arg(inp, "rows", 5, min_value=1),
        vertical_video_columns=_optional_int_arg(inp, "vertical_video_columns"),
        vertical_video_rows=_optional_int_arg(inp, "vertical_video_rows"),
        spacing=_int_arg(inp, "spacing", 2, min_value=0),
        background_color=inp.args.get("background_color", "black").strip() or "black",
        no_header=_bool_arg(inp, "no_header", False),
        header_font=_optional_str_arg(inp, "header_font"),
        header_font_size=_int_arg(inp, "header_font_size", 14, min_value=1),
        header_font_color=inp.args.get("header_font_color", "white").strip() or "white",
        timestamp_font=_optional_str_arg(inp, "timestamp_font"),
        timestamp_font_size=_int_arg(inp, "timestamp_font_size", 12, min_value=1),
        timestamp_font_color=inp.args.get("timestamp_font_color", "white").strip() or "white",
        timestamp_shadow_color=_shadow_color_arg(inp),
        comment_label=inp.args.get("comment_label", "Comment:").strip() or "Comment:",
        comment_text=_optional_str_arg(inp, "comment_text"),
        skip_seconds=_float_arg(inp, "skip_seconds", 10.0, min_value=0.0),
        jpeg_quality=_int_arg(inp, "jpeg_quality", 95, min_value=1),
    )


def resolve_font_path(font_arg: str | None, repo_root: Path) -> Path | None:
    if not font_arg:
        return None
    for base in (repo_root, Path.cwd()):
        candidate = (base / font_arg).resolve()
        if candidate.is_file():
            return candidate
    path = Path(font_arg)
    if path.is_file():
        return path.resolve()
    return None
