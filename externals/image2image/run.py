"""$image2image — transform images using joined prompt."""

from __future__ import annotations

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    new_images: list[str] = []
    for img_link in inp.bundle.images:
        content = (
            f"[emulated $image2image]\n"
            f"prompt: {inp.prompt_text}\n"
            f"source: {img_link}\n"
        )
        link = ctx.new_link("images", ".png", content.encode("utf-8"))
        new_images.append(link)
    if not inp.bundle.images:
        link = ctx.new_link(
            "images",
            ".png",
            f"[emulated $image2image]\n{inp.prompt_text}\n",
        )
        new_images.append(link)
    out.images = new_images
    return out
