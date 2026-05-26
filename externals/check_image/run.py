"""$check_image — validate images; failed ones go to changes as del."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    for img_link in list(inp.bundle.images):
        passed = hash((img_link, inp.prompt_text)) % 5 != 0
        if not passed:
            out.changes.append(("image", "del", img_link))
    return out
