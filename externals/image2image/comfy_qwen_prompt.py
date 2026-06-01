"""Shared Qwen Image Edit prompt template + Comfy-aligned template trim."""

from __future__ import annotations

import os
from typing import Any

# Comfy comfy_extras/nodes_qwen.py TextEncodeQwenImageEditPlus (2509 edit+).
QWEN_IMAGE_EDIT_LLAMA_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

PICTURE_PROMPT = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

_IM_START_TOKEN = 151644


def build_image_prompt_prefix(image_count: int) -> str:
    return "".join(PICTURE_PROMPT.format(i + 1) for i in range(image_count))


def format_qwen_edit_llama_text(image_prompt: str, user_prompt: str) -> str:
    """Full chat text passed to CLIP / processor (picture tokens + user instruction)."""
    return QWEN_IMAGE_EDIT_LLAMA_TEMPLATE.format(image_prompt + user_prompt)


def template_end_from_input_ids(input_ids: Any) -> int:
    """
    Match Comfy QwenImageTEModel.encode_token_weights template trim.

    Drops system + user wrapper tokens; keeps picture placeholders, vision
    expansion, and the user edit instruction.
    """
    row = input_ids[0]
    ids = row.tolist() if hasattr(row, "tolist") else list(row)
    template_end = 0
    count_im_start = 0
    for i, elem in enumerate(ids):
        if elem == _IM_START_TOKEN and count_im_start < 2:
            template_end = i
            count_im_start += 1

    if len(ids) > template_end + 3:
        if ids[template_end + 1] == 872 and ids[template_end + 2] == 198:
            template_end += 3

    return template_end


def debug_log_prompt_encode(*, user_prompt: str, llama_text: str, template_end: int, seq_len: int) -> None:
    if os.environ.get("AH_DEBUG_IMAGE2IMAGE", "").lower() not in ("1", "true", "yes"):
        return
    snippet = user_prompt.replace("\n", "\\n")[:120]
    print(
        f"$image2image: encode prompt={snippet!r} template_end={template_end} "
        f"embed_seq_len={seq_len}",
        flush=True,
    )
