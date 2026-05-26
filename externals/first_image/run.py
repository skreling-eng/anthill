"""$first_image — keep only the first link from the input images array."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.images = [inp.bundle.images[0]] if inp.bundle.images else []
    return out
