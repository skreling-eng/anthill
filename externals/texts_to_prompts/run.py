"""$texts_to_prompts — move input texts array into output prompts array."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    join = inp.args.get("join", "").lower() in ("1", "true", "yes")

    parts: list[str] = []
    for link in inp.bundle.texts:
        text = ctx.read_link_text(link)
        if not text:
            continue
        if join:
            parts.append(text)
        else:
            out.prompts.append(ctx.new_link("prompts", ".txt", text + "\n"))

    if join and parts:
        out.prompts.append(
            ctx.new_link("prompts", ".txt", "\n\n".join(parts) + "\n")
        )

    out.texts.clear()
    return out
