"""Estimate and trim prompts to fit GGUF context windows."""

from __future__ import annotations

import copy
import json
from typing import Any

# Code/JSON tends to tokenize slightly denser than plain prose.
_CHARS_PER_TOKEN = 3.5
_TEMPLATE_RESERVE_TOKENS = 256
_TRUNC_SUFFIX = "\n...[truncated by $code]...\n"


def estimate_tokens(text: str, *, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token) + 1)


def prompt_token_budget(
    n_ctx: int,
    max_tokens: int,
    *,
    reserve: int = _TEMPLATE_RESERVE_TOKENS,
) -> int:
    return max(512, n_ctx - max_tokens - reserve)


def prompt_char_budget(
    n_ctx: int,
    max_tokens: int,
    *,
    reserve: int = _TEMPLATE_RESERVE_TOKENS,
    chars_per_token: float = _CHARS_PER_TOKEN,
) -> int:
    return int(prompt_token_budget(n_ctx, max_tokens, reserve=reserve) * chars_per_token)


def auto_n_ctx(
    prompt_tokens: int,
    max_tokens: int,
    *,
    min_ctx: int = 4096,
    max_ctx: int = 131_072,
    reserve: int = _TEMPLATE_RESERVE_TOKENS,
) -> int:
    """Pick the smallest power-of-two context that fits prompt + generation + reserve."""
    need = prompt_tokens + max_tokens + reserve
    ctx = max(min_ctx, 512)
    while ctx < need and ctx < max_ctx:
        ctx *= 2
    return min(max(ctx, min_ctx), max_ctx)


def trim_plain_text(
    text: str,
    *,
    budget_chars: int,
    label: str = "prompt",
) -> tuple[str, list[str]]:
    """Truncate plain text with a trailing notice."""
    notes: list[str] = []
    if len(text) <= budget_chars:
        return text, notes
    keep = max(256, budget_chars - 64)
    notes.append(
        f"{label} truncated from {len(text)} to {keep} characters "
        f"(context budget {budget_chars} chars)"
    )
    return text[:keep] + "\n...[truncated by $llm]...\n", notes


def trim_code_request(
    request: dict[str, Any],
    *,
    budget_chars: int,
) -> tuple[dict[str, Any], list[str]]:
    """
    Shrink a $code JSON request to fit the input budget.

    Keeps prompts; trims file contents, then code_context; drops files if needed.
    """
    notes: list[str] = []
    req = copy.deepcopy(request)

    seen: set[str] = set()
    prompts: list[str] = []
    for item in req.get("prompts") or []:
        text = str(item)
        if text not in seen:
            seen.add(text)
            prompts.append(text)
    dropped_dupes = len(req.get("prompts") or []) - len(prompts)
    if dropped_dupes:
        notes.append(f"removed {dropped_dupes} duplicate prompt(s)")
    req["prompts"] = prompts

    def payload_size() -> int:
        return len(json.dumps(req, ensure_ascii=False))

    if payload_size() <= budget_chars:
        return req, notes

    req["_truncated"] = True

    files: list[dict[str, str]] = list(req.get("files") or [])
    while files and payload_size() > budget_chars:
        files.sort(key=lambda f: len(f.get("content", "")), reverse=True)
        largest = files[0]
        content = largest.get("content", "")
        name = largest.get("name", "?")
        if len(content) > 500:
            new_len = max(500, len(content) // 2)
            largest["content"] = content[:new_len] + _TRUNC_SUFFIX
            notes.append(f"truncated file {name!r} to {new_len} chars")
            continue
        files.pop(0)
        req["files"] = files
        notes.append(f"omitted file {name!r} (context limit)")

    ctx = str(req.get("code_context") or "")
    while ctx and payload_size() > budget_chars:
        new_len = max(256, len(ctx) // 2)
        ctx = ctx[:new_len] + _TRUNC_SUFFIX
        req["code_context"] = ctx
        notes.append("truncated code_context")

    if payload_size() > budget_chars:
        req["files"] = []
        req["code_context"] = ""
        notes.append("cleared files and code_context to fit context window")

    return req, notes
