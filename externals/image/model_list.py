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
        lora="lora/Pleasure.safetensors",
        desc="Crazy Desire",
        poshelp="Big smile, crazy pleasure",
    ),
    ImageGen("crazy_desire_realistic", gentype="flux", 
        loras=[
           'lora/Pleasure.safetensors',
           'lora/realistic effect.safetensors',
        ], desc="", poshelp="Big smile, crazy pleasure, realistic effect, oil, high-quality detail"),
    ImageGen(
        "crazy_desire_realistic2",
        gentype="flux_ext",
        model="flux/fluxFusionV24StepsGGUFNF4_V2NF4AIO.safetensors",
        lora="lora/Pleasure.safetensors",
        loras=[
           'lora/Pleasure.safetensors',
           'lora/realistic effect.safetensors',
        ],
        desc="Crazy Desire",
        poshelp="Big smile, crazy pleasure",
    ),
    ImageGen(
        "flux_fusion_v2",
        gentype="flux_ext",
        model="flux/fluxFusionV24StepsGGUFNF4_V2NF4AIO.safetensors",
        steps=4,
        guidance_scale=3.5,
        desc="Flux Fusion V2 NF4 AIO (Comfy checkpoint under models/flux/)",
    ),
    ImageGen(
        "flux_fusion_v2_fp16",
        gentype="flux_ext",
        model="flux/fluxFusionV24StepsGGUFNF4_v1Fp16AIO.safetensors",
        steps=4,
        guidance_scale=3.5,
        desc="Flux Fusion V2 (needs NF4/GGUF file on ≤16 GiB GPUs; Fp16AIO is ~27 GiB)",
    ),
    ImageGen(
        "flux2_klein_fp8",
        gentype="flux2_klein",
        model="flux2klein/flux2Klein9bFp8_fp8.safetensors",
        steps=4,
        guidance_scale=1.0,
        desc="Flux.2 Klein 9B FP8 distilled (4 steps, cfg=1; use base checkpoint for 20 steps)",
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
