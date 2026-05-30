"""TextEncodeQwenImageEditPlus (Comfy comfy_extras/nodes_qwen.py logic, no comfy_api)."""

from __future__ import annotations

import math


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
    llama_template = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, "
        "objects, background), then explain how the user's text instruction should "
        "alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
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
            ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_latents:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": ref_latents}
        )
    return conditioning
