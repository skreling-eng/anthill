"""$folder — load all files from a directory into arrays by extension."""

from __future__ import annotations

import fnmatch
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
  @py_files: $folder('src', '*.py')
  @sources: $folder('ahlib', source_path=True)

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


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _source_path_from_args(args: dict[str, str]) -> bool:
    return _truthy(args.get("source_path", ""))


def _pattern_from_args(args: dict[str, str]) -> str:
    return (
        args.get("pattern", "")
        or args.get("_pattern", "")
        or args.get("_path2", "")
    ).strip()


def _matches_pattern(name: str, pattern: str) -> bool:
    if not pattern:
        return True
    return fnmatch.fnmatch(name, pattern)


def _list_files(dir_path: Path, pattern: str = "") -> list[Path]:
    return sorted(
        p
        for p in dir_path.iterdir()
        if p.is_file() and _matches_pattern(p.name, pattern)
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    path = inp.args.get("_path", inp.args.get("path", "")).strip()
    pattern = _pattern_from_args(inp.args)
    source_path = _source_path_from_args(inp.args)
    if not path:
        raise RuntimeError(_HELP.strip())

    if _emulate_enabled():
        try:
            dir_path = _resolve_dir_path(ctx, path)
        except FileNotFoundError:
            dir_path = (_REPO_ROOT / path).resolve()
        for name in ("sample.mp4", "sample.wav", "sample.png", "readme.txt"):
            if not _matches_pattern(name, pattern):
                continue
            ext = Path(name).suffix.lower()
            arr_key = _EXT_ARRAY.get(ext, "files")
            if source_path:
                link = str((dir_path / name).resolve()).replace("\\", "/")
            else:
                content = f"[emulated folder: {path}]\nfile: {name}\n"
                link = ctx.new_link(arr_key, ext or ".bin", content)
            getattr(out, arr_key).append(link)
        return out

    dir_path = _resolve_dir_path(ctx, path)
    for src in _list_files(dir_path, pattern):
        ext = src.suffix.lower()
        arr_key = _EXT_ARRAY.get(ext, "files")
        if source_path:
            link = str(src.resolve()).replace("\\", "/")
        else:
            link = ctx.new_link(arr_key, ext or src.suffix or ".bin", src.read_bytes())
        getattr(out, arr_key).append(link)
    return out
