"""$json2texts — split JSON array in texts[] into one text link per element."""

from __future__ import annotations

import json

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_FALLBACK_KEYS = ("text", "snippet", "description", "title", "content", "url")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _parse_array(raw: str, *, source: str) -> list:
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"$json2texts: invalid JSON in {source}: {exc}") from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "items", "data", "rows"):
            nested = data.get(key)
            if isinstance(nested, list):
                return nested
    raise ValueError(
        f"$json2texts: expected JSON array in {source}, got {type(data).__name__}"
    )


def _element_text(elem: object, *, field: str) -> str:
    if isinstance(elem, str):
        return elem
    if isinstance(elem, (int, float, bool)) or elem is None:
        return "" if elem is None else str(elem)
    if isinstance(elem, dict):
        if field and field in elem and elem[field] is not None:
            return str(elem[field]).strip()
        for key in (field,) + _FALLBACK_KEYS if field else _FALLBACK_KEYS:
            if key and key in elem and elem[key] is not None:
                val = str(elem[key]).strip()
                if val:
                    return val
        return json.dumps(elem, ensure_ascii=False, indent=2)
    return json.dumps(elem, ensure_ascii=False, indent=2)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()

    field = inp.args.get("field", "text").strip()
    join = _truthy(inp.args.get("join", ""))
    parts: list[str] = []

    for link in inp.bundle.texts:
        raw = ctx.read_link_text(link)
        for elem in _parse_array(raw, source=link):
            text = _element_text(elem, field=field)
            if not text:
                continue
            if join:
                parts.append(text)
            else:
                out.texts.append(ctx.new_link("texts", ".txt", text + "\n"))

    if join and parts:
        out.texts.append(ctx.new_link("texts", ".txt", "\n\n".join(parts) + "\n"))

    return out
