"""$select — pick elements from input arrays by index lists."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_parser import ARRAY_TYPES
from ahlib.ah_runtime import ArrayBundle

_ARRAY_ALIASES: dict[str, str] = {
    "text": "texts",
    "prompt": "prompts",
    "image": "images",
    "sound": "sounds",
    "video": "videos",
    "file": "files",
}


def _array_key(raw: str) -> str:
    key = _ARRAY_ALIASES.get(raw, raw)
    if key not in ARRAY_TYPES or key == "changes":
        raise ValueError(f"$select invalid array name: {raw!r}")
    return key


def _parse_index_token(raw: str) -> int:
    token = raw.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1].strip()
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"$select: invalid index {raw!r}") from exc


def _parse_index_list(raw: str) -> list[int]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [_parse_index_token(part) for part in text.split(",") if part.strip()]


def _index_list(inp: ExternalInput, key: str) -> list[int]:
    if key in inp.arg_lists and inp.arg_lists[key]:
        return [_parse_index_token(x) for x in inp.arg_lists[key]]
    if key in inp.args and inp.args[key]:
        return _parse_index_list(inp.args[key])
    raise ValueError(f"$select: missing index list for {key!r}")


def _select_specs(inp: ExternalInput) -> dict[str, list[int]]:
    keys = {
        key
        for key in set(inp.args) | set(inp.arg_lists)
        if not key.startswith("_")
    }
    if not keys:
        raise ValueError(
            "$select requires array=index lists, e.g. $select(sounds=[1])"
        )
    specs: dict[str, list[int]] = {}
    for key in sorted(keys):
        array_key = _array_key(key)
        specs[array_key] = _index_list(inp, key)
    return specs


def _pick_links(links: list[str], indexes: list[int], key: str) -> list[str]:
    out: list[str] = []
    for index in indexes:
        if index < 0 or index >= len(links):
            raise IndexError(
                f"$select: {key}[{index}] out of range (len={len(links)})"
            )
        out.append(links[index])
    return out


def run(_ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    specs = _select_specs(inp)
    out = ArrayBundle()
    for key, indexes in specs.items():
        links = getattr(inp.bundle, key)
        getattr(out, key).extend(_pick_links(links, indexes, key))
    return out
