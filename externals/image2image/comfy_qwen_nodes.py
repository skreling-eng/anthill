"""TextEncodeQwenImageEditPlus (Comfy comfy_extras/nodes_qwen.py logic, no comfy_api)."""

from __future__ import annotations

import math

from externals.image2image.comfy_qwen_prompt import (
    QWEN_IMAGE_EDIT_LLAMA_TEMPLATE,
    debug_log_prompt_encode,
)

_EMPTY_NEGATIVE_CACHE: dict[int, object] = {}


def fast_empty_negative_conditioning(clip) -> object:
    """Workflow negative node has prompt='' — cache result (cfg=1, no vision)."""
    key = id(getattr(clip, "patcher", clip))
    cached = _EMPTY_NEGATIVE_CACHE.get(key)
    if cached is not None:
        return cached
    tokens = clip.tokenize("", images=[], llama_template=QWEN_IMAGE_EDIT_LLAMA_TEMPLATE)
    cached = clip.encode_from_tokens_scheduled(tokens)
    _EMPTY_NEGATIVE_CACHE[key] = cached
    return cached


def text_encode_qwen_image_edit(
    clip,
    prompt: str,
    vae=None,
    *,
    image=None,
):
    """Comfy TextEncodeQwenImageEdit — single source image + reference_latents."""
    import comfy.utils
    import node_helpers

    ref_latent = None
    images: list = []
    if image is not None:
        samples = image.movedim(-1, 1)
        total = int(1024 * 1024)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        image = s.movedim(1, -1)
        images = [image[:, :, :, :3]]
        if vae is not None:
            pixels = image[:, :, :, :3]
            from externals.comfy_inprocess.comfy_memory import load_vae_for_encode

            load_vae_for_encode(
                vae,
                length=int(pixels.shape[0]),
                height=int(pixels.shape[1]),
                width=int(pixels.shape[2]),
            )
            ref_latent = vae.encode(pixels)

    tokens = clip.tokenize(prompt, images=images)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_latent is not None:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": [ref_latent]}, append=True
        )
    return conditioning


def text_encode_qwen_image_edit_plus(
    clip,
    prompt: str,
    vae=None,
    *,
    image1=None,
    image2=None,
    image3=None,
):
    import comfy.utils
    import node_helpers

    ref_latents = []
    images = [image1, image2, image3]
    images_vl = []
    llama_template = QWEN_IMAGE_EDIT_LLAMA_TEMPLATE
    image_prompt = ""

    for i, image in enumerate(images):
        if image is None:
            continue
        samples = image.movedim(-1, 1)
        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        images_vl.append(s.movedim(1, -1))
        if vae is not None:
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by / 8.0) * 8
            height = round(samples.shape[2] * scale_by / 8.0) * 8
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            pixels = s.movedim(1, -1)[:, :, :, :3]
            from externals.comfy_inprocess.comfy_memory import load_vae_for_encode

            load_vae_for_encode(
                vae,
                length=int(pixels.shape[0]),
                height=int(pixels.shape[1]),
                width=int(pixels.shape[2]),
            )
            ref_latents.append(vae.encode(pixels))
        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
            i + 1
        )

    llama_text = llama_template.format(image_prompt + prompt)
    debug_log_prompt_encode(
        user_prompt=prompt,
        llama_text=llama_text,
        template_end=-1,
        seq_len=len(llama_text),
    )
    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_latents:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": ref_latents}, append=True
        )
    return conditioning
