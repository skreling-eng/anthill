"""$search — web search via DuckDuckGo; results as JSON in texts[]."""

from __future__ import annotations

import json
import os

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.search.duckduckgo import search
from externals.search.page_fetch import (
    enrich_results_with_pages,
    parse_fetch_pages_arg,
)
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    for key in ("AH_EMULATE_SEARCH", "AH_EMULATE_SERCH"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _queries(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    queries = read_prompt_texts(ctx, inp)
    return [q for q in queries if q.strip()]


def _emulate_results(
    query: str, *, limit: int, fetch_pages: bool
) -> list[dict[str, str]]:
    row: dict[str, str] = {
        "url": f"https://example.com/search?q={query.replace(' ', '+')}",
        "text": f"[emulated $search] Result for: {query}",
    }
    if fetch_pages:
        page = (
            f"[emulated page] Article about {query}. "
            "Main facts would appear here after fetch_pages."
        )
        row["page_text"] = page
        row["page_fetch"] = "ok"
        row["text"] = f"{row['text']}\n\n--- page ---\n{page}"
    return [row][:limit]


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    limit = max(1, min(int(inp.args.get("limit", "10")), 100))
    lang = inp.args.get("lang", "")
    region = inp.args.get("region", "")
    site = inp.args.get("site", "")
    timeout = float(inp.args.get("timeout", "60"))
    fetch_pages = parse_fetch_pages_arg(inp.args.get("fetch_pages", ""))
    fetch_max = max(0, min(int(inp.args.get("fetch_max", "3")), limit))
    fetch_timeout = float(inp.args.get("fetch_timeout", "15"))
    fetch_max_chars = max(500, min(int(inp.args.get("fetch_max_chars", "12000")), 100_000))

    queries = _queries(ctx, inp)
    if not queries:
        queries = ["(empty query)"]

    for query in queries:
        if _emulate_enabled():
            rows = _emulate_results(query, limit=limit, fetch_pages=fetch_pages)
        else:
            rows = search(
                query,
                limit=limit,
                lang=lang,
                region=region,
                site=site,
                timeout=timeout,
            )
            if fetch_pages and rows:
                rows = enrich_results_with_pages(
                    rows,
                    max_pages=fetch_max,
                    timeout=fetch_timeout,
                    max_chars=fetch_max_chars,
                )
        payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        link = ctx.new_link("texts", ".txt", payload)
        out.texts.append(link)

    out.prompts.clear()
    return out
