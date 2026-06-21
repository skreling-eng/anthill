"""Estimate and trim prompts to fit GGUF context windows (brain-local)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

CHARS_PER_TOKEN = 3.5
TEMPLATE_RESERVE_TOKENS = 256
TRUNC_SUFFIX = "\n...[truncated by brain]...\n"

ChatHistory = list[tuple[str, str]]


def trim_chat_history(
    history: ChatHistory,
    *,
    budget_chars: int,
) -> tuple[ChatHistory, list[str]]:
    """Drop or shorten oldest turns until history fits the budget."""
    notes: list[str] = []
    pairs: ChatHistory = list(history)
    if not pairs or budget_chars <= 0:
        return [], notes

    def total_size() -> int:
        return sum(len(u) + len(a) + 24 for u, a in pairs)

    while pairs and total_size() > budget_chars:
        user, assistant = pairs[0]
        if len(assistant) > 500:
            new_len = len(assistant) // 2
            candidate = assistant[:new_len] + TRUNC_SUFFIX
            if len(candidate) >= len(assistant):
                pairs.pop(0)
                notes.append("dropped oldest conversation turn (context limit)")
            else:
                pairs[0] = (user, candidate)
                notes.append("truncated older assistant reply (context limit)")
            continue
        pairs.pop(0)
        notes.append("dropped oldest conversation turn (context limit)")

    return pairs, notes


def estimate_messages_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m.get("content", "")) + 16 for m in messages)


def build_chat_messages(
    prompt: str,
    *,
    system: str = "",
    history: ChatHistory | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    for user_msg, assistant_msg in history or []:
        if user_msg.strip():
            messages.append({"role": "user", "content": user_msg.strip()})
        if assistant_msg.strip():
            messages.append({"role": "assistant", "content": assistant_msg.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def estimate_tokens(text: str, *, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token) + 1)


def prompt_token_budget(
    n_ctx: int,
    max_tokens: int,
    *,
    reserve: int = TEMPLATE_RESERVE_TOKENS,
) -> int:
    return max(512, n_ctx - max_tokens - reserve)


def prompt_char_budget(
    n_ctx: int,
    max_tokens: int,
    *,
    reserve: int = TEMPLATE_RESERVE_TOKENS,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> int:
    return int(prompt_token_budget(n_ctx, max_tokens, reserve=reserve) * chars_per_token)


def auto_n_ctx(
    prompt_tokens: int,
    max_tokens: int,
    *,
    min_ctx: int = 4096,
    max_ctx: int = 131_072,
    reserve: int = TEMPLATE_RESERVE_TOKENS,
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
    notes: list[str] = []
    if len(text) <= budget_chars:
        return text, notes
    keep = max(256, budget_chars - len(TRUNC_SUFFIX) - 16)
    notes.append(
        f"{label} truncated from {len(text)} to {keep} characters "
        f"(budget {budget_chars} chars)"
    )
    return text[:keep] + TRUNC_SUFFIX, notes


@dataclass
class AgentPromptParts:
    request: str
    conversation: str = ""
    plan_summary: str = ""
    tree: str = ""
    context_files: dict[str, str] = field(default_factory=dict)
    grep_hits: list[tuple[str, int, str]] = field(default_factory=list)
    search_results: list[dict[str, str]] = field(default_factory=list)
    footer: str = "Produce analysis and unified diffs for the requested change."

    def render(self) -> str:
        file_blocks = [
            f"### File: {path}\n```\n{content}\n```"
            for path, content in self.context_files.items()
        ]
        grep_lines = [
            f"{path}:{lineno}: {line}" for path, lineno, line in self.grep_hits
        ]
        search_lines = [
            f"- [{r.get('query', '?')}] {r.get('url', '')}\n  {r.get('text', '')[:400]}"
            for r in self.search_results
        ]
        parts = [
            f"# Prior conversation\n{self.conversation}\n" if self.conversation else "",
            f"# User change request\n{self.request}\n",
            f"# Plan summary\n{self.plan_summary}\n" if self.plan_summary else "",
            f"# File tree (abbreviated)\n{self.tree}\n" if self.tree else "",
            "# Source files\n" + "\n\n".join(file_blocks) + "\n" if file_blocks else "",
            "# Grep hits\n" + "\n".join(grep_lines) + "\n" if grep_lines else "",
            "# Web search results\n" + "\n".join(search_lines) + "\n"
            if search_lines
            else "",
            self.footer,
        ]
        return "\n".join(p for p in parts if p.strip())

    def size(self) -> int:
        return len(self.render())


def trim_agent_prompt(
    parts: AgentPromptParts,
    *,
    budget_chars: int,
) -> tuple[AgentPromptParts, list[str]]:
    """
    Shrink an agent prompt to fit the input budget.

    Order: tree → search → grep → file contents (largest first) → drop files.
    """
    notes: list[str] = []
    p = copy.deepcopy(parts)

    def fits() -> bool:
        return p.size() <= budget_chars

    if fits():
        return p, notes

    if p.conversation and not fits():
        p.conversation, conv_notes = trim_plain_text(
            p.conversation,
            budget_chars=min(len(p.conversation), 4000),
            label="conversation",
        )
        notes.extend(conv_notes)

    if p.tree:
        p.tree, tree_notes = trim_plain_text(p.tree, budget_chars=min(len(p.tree), 6000), label="tree")
        notes.extend(tree_notes)

    while p.search_results and not fits():
        p.search_results.pop()
        notes.append("dropped web search result (context limit)")

    while len(p.grep_hits) > 5 and not fits():
        p.grep_hits.pop()
        notes.append("dropped grep hit (context limit)")

    files = list(p.context_files.items())
    while files and not fits():
        files.sort(key=lambda item: len(item[1]), reverse=True)
        path, content = files[0]
        if len(content) > 800:
            new_len = len(content) // 2
            candidate = content[:new_len] + TRUNC_SUFFIX
            if len(candidate) >= len(content):
                del p.context_files[path]
                notes.append(f"omitted file {path!r} (context limit)")
            else:
                p.context_files[path] = candidate
                notes.append(f"truncated {path!r} to {new_len} chars")
            files = list(p.context_files.items())
            continue
        del p.context_files[path]
        files = list(p.context_files.items())
        notes.append(f"omitted file {path!r} (context limit)")

    if not fits():
        p.tree, tree_notes = trim_plain_text(
            p.tree, budget_chars=2000, label="tree (final)"
        )
        notes.extend(tree_notes)

    if not fits():
        prompt, prompt_notes = trim_plain_text(
            p.render(), budget_chars=budget_chars, label="full prompt"
        )
        notes.extend(prompt_notes)
        return AgentPromptParts(request=prompt, footer=""), notes

    return p, notes


def trim_code_request(
    request: dict[str, Any],
    *,
    budget_chars: int,
) -> tuple[dict[str, Any], list[str]]:
    """Shrink a JSON code request (planner payloads) to fit the input budget."""
    notes: list[str] = []
    req = copy.deepcopy(request)

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
            largest["content"] = content[:new_len] + TRUNC_SUFFIX
            notes.append(f"truncated file {name!r} to {new_len} chars")
            continue
        files.pop(0)
        req["files"] = files
        notes.append(f"omitted file {name!r} (context limit)")

    ctx = str(req.get("code_context") or "")
    while ctx and payload_size() > budget_chars:
        new_len = max(256, len(ctx) // 2)
        ctx = ctx[:new_len] + TRUNC_SUFFIX
        req["code_context"] = ctx
        notes.append("truncated code_context")

    return req, notes
