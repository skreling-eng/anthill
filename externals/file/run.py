"""$file — load a file from disk into the proper array by extension."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXT_ARRAY = {
    ".mp3": "sounds",
    ".wav": "sounds",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
    ".webp": "images",
    ".mp4": "videos",
    ".txt": "texts",
}


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_FILE", "").lower() in ("1", "true", "yes")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _source_path_from_args(args: dict[str, str]) -> bool:
    return _truthy(args.get("source_path", ""))


def _resolve_source_path(ctx: ExternalContext, ref: str) -> Path:
    raw = Path(ref)
    direct: list[Path] = []
    if raw.is_absolute():
        direct.append(raw)
    direct.extend(
        [
            Path.cwd() / raw,
            _REPO_ROOT / raw,
            ctx.base_dir / raw,
            ctx.base_dir.parent / raw,
        ]
    )
    seen: set[Path] = set()
    for path in direct:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()

    tried = ", ".join(str(p) for p in direct[:6])
    raise FileNotFoundError(f"$file: not found: {ref!r} (tried {tried})")


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    path = inp.args.get("_path", inp.args.get("path", "unknown.bin"))
    ext = Path(path).suffix.lower()
    arr_key = _EXT_ARRAY.get(ext, "files")
    source_path = _source_path_from_args(inp.args)

    if _emulate_enabled():
        if source_path:
            try:
                src = _resolve_source_path(ctx, path)
                link = str(src.resolve()).replace("\\", "/")
            except FileNotFoundError:
                link = ctx.new_link(arr_key, ext or ".bin", f"[emulated file: {path}]\n")
        else:
            link = ctx.new_link(arr_key, ext or ".bin", f"[emulated file: {path}]\n")
        getattr(out, arr_key).append(link)
        return out

    src = _resolve_source_path(ctx, path)
    if source_path:
        link = str(src.resolve()).replace("\\", "/")
    else:
        link = ctx.new_link(arr_key, ext or src.suffix or ".bin", src.read_bytes())
    getattr(out, arr_key).append(link)
    return out
