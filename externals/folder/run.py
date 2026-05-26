"""$folder — load all files from a directory into arrays by extension."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.file.run import _EXT_ARRAY
from ahlib.ah_runtime import ArrayBundle

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HELP = """
$folder needs a directory path.

Example:
  @videos: $folder('test_data/videos')
  @assets: $folder('test_data/mixed')

Loads every file in the directory (non-recursive, sorted by name) into the
array matching its extension — same rules as $file. Set AH_EMULATE_FOLDER=1
for stub links without reading disk.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_FOLDER", "").lower() in ("1", "true", "yes")


def _resolve_dir_path(ctx: ExternalContext, ref: str) -> Path:
    raw = Path(ref)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            Path.cwd() / raw,
            _REPO_ROOT / raw,
            ctx.base_dir / raw,
            ctx.base_dir.parent / raw,
        ]
    )

    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates[:6])
    raise FileNotFoundError(f"$folder: directory not found: {ref!r} (tried {tried})")


def _list_files(dir_path: Path) -> list[Path]:
    return sorted(p for p in dir_path.iterdir() if p.is_file())


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    path = inp.args.get("_path", inp.args.get("path", "")).strip()
    if not path:
        raise RuntimeError(_HELP.strip())

    if _emulate_enabled():
        for name in ("sample.mp4", "sample.wav", "sample.png", "readme.txt"):
            ext = Path(name).suffix.lower()
            arr_key = _EXT_ARRAY.get(ext, "files")
            content = f"[emulated folder: {path}]\nfile: {name}\n"
            link = ctx.new_link(arr_key, ext or ".bin", content)
            getattr(out, arr_key).append(link)
        return out

    dir_path = _resolve_dir_path(ctx, path)
    for src in _list_files(dir_path):
        ext = src.suffix.lower()
        arr_key = _EXT_ARRAY.get(ext, "files")
        link = ctx.new_link(arr_key, ext or src.suffix or ".bin", src.read_bytes())
        getattr(out, arr_key).append(link)
    return out
