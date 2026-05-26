"""$list — split input prompts into texts (comma / newline separated)."""

from __future__ import annotations

import re

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from ahlib.ah_runtime import ArrayBundle

_SPLIT_RE = re.compile(r"[,\n]+")


def _split_items(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()

    for prompt in read_prompt_texts(ctx, inp):
        for item in _split_items(prompt):
            out.texts.append(ctx.new_link("texts", ".txt", item + "\n"))

    out.prompts.clear()
    return out
