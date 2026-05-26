"""Fallback handler for unknown $ externals."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput, *, name: str = "unknown") -> ArrayBundle:
    out = inp.bundle.copy()
    link = ctx.new_link("texts", ".txt", f"[emulated ${name}]\n")
    out.texts.append(link)
    return out
