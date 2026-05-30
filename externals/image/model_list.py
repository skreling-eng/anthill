"""Named image generation profiles for $image(model='...')."""

from __future__ import annotations

from externals.image.fastimage import ImageGen

imgens_list = [
    ImageGen("default", gentype="flux", lora=""),
    ImageGen(
        "realitsic_fantasy",
        gentype="flux",
        lora="lora/FluxMythR3alisticF.safetensors",
    ),
    ImageGen(
        "alien_abduction",
        gentype="flux",
        lora="lora/AlienAbduction01-00_CE_FLUX_AIT.safetensors",
        poshelp="alnabdctnCE style",
    ),
    ImageGen(
        "aibeauty01",
        gentype="flux",
        lora="lora/aibeauty01flux_20.safetensors",
        poshelp="AI beauty",
    ),
    ImageGen(
        "new_fantasy_corev4",
        gentype="flux",
        lora="lora/New_Fantasy_CoreV4_FLUX.safetensors",
    ),
    ImageGen(
        "meme_peter_face",
        gentype="flux",
        lora="lora/opt-meme_peter_face_v0_rank4_bf16.safetensors",
    ),
    ImageGen(
        "linedesign_flux",
        gentype="flux",
        lora="lora/linedesign_flux.safetensors",
    ),
    ImageGen(
        "crazy_librarian",
        gentype="flux",
        lora="lora/DonM__Crazy_Librarian_Character_Flux-000001.safetensors",
    ),
    ImageGen(
        "crazy_desire",
        gentype="flux",
        lora="lora/DonM__Crazy_Librarian_Character_Flux-000001.safetensors",
        desc="alias for crazy_librarian-style generation",
    ),
]

imgens: dict[str, ImageGen] = {g.name: g for g in imgens_list}


def get_image_gen(name: str) -> ImageGen:
    """Resolve model name from $image(model='name')."""
    if name in imgens:
        return imgens[name]
    if name == "default" or not name:
        return imgens["default"]
    available = ", ".join(sorted(imgens))
    raise KeyError(f"Unknown image model {name!r}. Available: {available}")
