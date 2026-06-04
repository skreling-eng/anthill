"""Strip leaked LLM chat-template tokens from user-visible text."""

from __future__ import annotations

import re

# Literal tokens sometimes emitted by Gemma, Qwen, Llama chat templates.
_LITERAL_TOKENS = (
    "<end_of_turn>",
    "<start_of_turn>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
)

# <start_of_turn>user, <end_of_turn>, <|...|>, etc.
_TAG_RE = re.compile(
    r"<\|[^>\n|]*\|>"
    r"|<\/?(?:start_of_turn|end_of_turn)(?:\s[^>\n]*)?>"
    r"|<\/?(?:im_start|im_end)(?:\s[^>\n]*)?>",
    re.IGNORECASE,
)
_IM_START_ROLE_RE = re.compile(
    r"<\|im_start\|>\s*(?:assistant|user|system|bot)\s*\n?",
    re.IGNORECASE,
)
_LEADING_ROLE_LINE_RE = re.compile(
    r"^(?:assistant|user|system|model|bot)\s*\n",
    re.IGNORECASE,
)


def clean_display_text(text: str) -> str:
    """Remove special chat tokens; collapse extra blank lines."""
    if not text:
        return text
    out = text
    out = _IM_START_ROLE_RE.sub("", out)
    for tok in _LITERAL_TOKENS:
        out = out.replace(tok, "")
    out = _TAG_RE.sub("", out)
    out = _LEADING_ROLE_LINE_RE.sub("", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
