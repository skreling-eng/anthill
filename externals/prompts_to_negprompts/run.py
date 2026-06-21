"""$prompts_to_negprompts — move input prompts array into output negprompts array."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    join = inp.args.get("join", "").lower() in ("1", "true", "yes")

    parts: list[str] = []
    for link in inp.bundle.prompts:
        text = ctx.read_link_text(link)
        if not text:
            continue
        if join:
            parts.append(text)
        else:
            out.negprompts.append(ctx.new_link("negprompts", ".txt", text + "\n"))

    if join and parts:
        out.negprompts.append(
            ctx.new_link("negprompts", ".txt", "\n\n".join(parts) + "\n")
        )

    out.prompts.clear()
    return out
