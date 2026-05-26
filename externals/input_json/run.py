"""$input_json — load a bundle manifest (output.json) into session arrays."""

from __future__ import annotations

import json
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_parser import ARRAY_TYPES
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$input_json loads array links from a JSON file (same shape as output.json).

When the JSON file lives inside a session folder, each link is resolved to an
absolute path under that session (so assets work even from another run session).

Example:
  @restore: $input_json('sessions/20260519_051548_2592/11__music/output.json')

Optional: strict=1 — fail if any resolved file is missing.
"""


def _resolve_json_path(ctx: ExternalContext, ref: str) -> Path:
    raw = Path(ref)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(ctx.base_dir / raw)
    candidates.append(Path.cwd() / raw)
    candidates.append(raw.resolve())

    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates[:3])
    raise FileNotFoundError(f"$input_json: JSON not found: {ref!r} (tried {tried})")


def _source_session_root(json_path: Path, ctx: ExternalContext) -> Path | None:
    """Session directory that owns the JSON manifest and its linked files."""
    resolved = json_path.resolve()
    try:
        resolved.relative_to(ctx.base_dir.resolve())
        return ctx.base_dir.resolve()
    except ValueError:
        pass
    parts = resolved.parts
    if "sessions" in parts:
        i = parts.index("sessions")
        if i + 1 < len(parts):
            return Path(*parts[: i + 2])
    return None


def _normalize_link(link: str) -> str:
    return link.replace("\\", "/").lstrip("./")


def _resolve_link(link: str, source_session: Path) -> str:
    normalized = _normalize_link(link)
    path = Path(normalized)
    if path.is_absolute():
        return str(path.resolve()).replace("\\", "/")
    absolute = (source_session / normalized).resolve()
    return str(absolute).replace("\\", "/")


def _load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"$input_json: invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"$input_json: root must be a JSON object in {path}")
    return data


def _bundle_from_manifest(data: dict, *, source_session: Path | None) -> ArrayBundle:
    payload: dict[str, list] = {}
    for key in ARRAY_TYPES:
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, list):
            raise ValueError(f"$input_json: {key!r} must be a JSON array")
        if key == "changes":
            payload[key] = value
            continue
        links = [_normalize_link(str(item)) for item in value]
        if source_session is not None:
            links = [_resolve_link(link, source_session) for link in links]
        payload[key] = links
    return ArrayBundle.from_dict(payload)


def _validate_links(ctx: ExternalContext, bundle: ArrayBundle) -> None:
    missing: list[str] = []
    for key in ARRAY_TYPES:
        if key == "changes":
            continue
        for link in getattr(bundle, key):
            path = Path(link)
            if not path.is_absolute():
                path = ctx.base_dir / link
            if not path.is_file():
                missing.append(link)
    if missing:
        sample = "\n  ".join(missing[:8])
        extra = f"\n  … and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"$input_json: {len(missing)} linked file(s) missing:\n  {sample}{extra}"
        )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    ref = inp.args.get("_path", inp.args.get("path", "")).strip()
    if not ref:
        raise RuntimeError(_HELP.strip())

    json_path = _resolve_json_path(ctx, ref)
    source_session = _source_session_root(json_path, ctx)
    bundle = _bundle_from_manifest(
        _load_manifest(json_path),
        source_session=source_session,
    )

    strict = inp.args.get("strict", "").lower() in ("1", "true", "yes")
    if strict:
        _validate_links(ctx, bundle)

    return bundle
