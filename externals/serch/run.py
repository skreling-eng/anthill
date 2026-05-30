"""$serch — web search via DuckDuckGo; results as JSON in texts[]."""

from __future__ import annotations

import json
import os

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.serch.duckduckgo import search
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_SERCH", "").lower() in ("1", "true", "yes")


def _queries(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    queries = read_prompt_texts(ctx, inp)
    return [q for q in queries if q.strip()]


def _emulate_results(query: str, *, limit: int) -> list[dict[str, str]]:
    return [
        {
            "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            "text": f"[emulated $serch] Result for: {query}",
        }
    ][:limit]


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    limit = max(1, min(int(inp.args.get("limit", "10")), 100))
    lang = inp.args.get("lang", "")
    region = inp.args.get("region", "")
    site = inp.args.get("site", "")
    timeout = float(inp.args.get("timeout", "60"))

    queries = _queries(ctx, inp)
    if not queries:
        queries = ["(empty query)"]

    for query in queries:
        if _emulate_enabled():
            rows = _emulate_results(query, limit=limit)
        else:
            rows = search(
                query,
                limit=limit,
                lang=lang,
                region=region,
                site=site,
                timeout=timeout,
            )
        payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        link = ctx.new_link("texts", ".txt", payload)
        out.texts.append(link)

    out.prompts.clear()
    return out
