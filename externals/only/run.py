"""$only — keep selected input arrays; drop all others."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_parser import ARRAY_TYPES
from ahlib.ah_runtime import ArrayBundle


def _array_keys(inp: ExternalInput) -> list[str]:
    raw = inp.args.get("_arrays", "").strip()
    if not raw:
        raise ValueError("$only(...) requires array names, e.g. $only(images, prompts)")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError("$only(...) requires at least one array name")
    for key in keys:
        if key not in ARRAY_TYPES or key == "changes":
            raise ValueError(f"$only(...) invalid array name: {key!r}")
    return keys


def run(_ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    keys = _array_keys(inp)
    out = ArrayBundle()
    for key in keys:
        getattr(out, key).extend(getattr(inp.bundle, key))
    return out
