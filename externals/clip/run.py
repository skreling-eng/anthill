"""$clip — combine images and sounds into a video."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    content = (
        f"[emulated $clip]\n"
        f"images: {inp.bundle.images}\n"
        f"sounds: {inp.bundle.sounds}\n"
    )
    link = ctx.new_link("videos", ".mp4", content)
    out.videos.append(link)
    return out
