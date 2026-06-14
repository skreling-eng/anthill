"""Comfy node handlers for $controlnet."""

from __future__ import annotations

from externals.comfy_inprocess.executor import register_node_handler
from externals.image2image.comfy_executor import execute_prompt_legacy  # noqa: F401
from externals.image2image.comfy_qwen_nodes import (
    fast_empty_negative_conditioning,
    text_encode_qwen_image_edit,
)

_CLIP_CACHE: dict[str, object] = {}


def _qwen_clip_loader(inputs: dict) -> tuple:
    import folder_paths
    import comfy.sd
    from comfy.sd import CLIPType

    clip_name = inputs["clip_name"]
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    key = str(clip_path)
    if key not in _CLIP_CACHE:
        print(f"$controlnet: loading CLIP {clip_name}…", flush=True)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=CLIPType.QWEN_IMAGE,
        )
        _CLIP_CACHE[key] = clip
    return (_CLIP_CACHE[key],)


def _set_union_controlnet_type(inputs: dict) -> tuple:
    from comfy.cldm.control_types import UNION_CONTROLNET_TYPES

    control_net = inputs["control_net"]
    union_type = inputs.get("type", "auto")
    control_net = control_net.copy()
    type_number = UNION_CONTROLNET_TYPES.get(union_type, -1)
    if type_number >= 0:
        control_net.set_extra_arg("control_type", [type_number])
    else:
        control_net.set_extra_arg("control_type", [])
    return (control_net,)


def _text_encode_source(inputs: dict) -> tuple:
    prompt = inputs.get("prompt", "") or ""
    if prompt.strip():
        print(f"$controlnet: encoding source-conditioned prompt ({len(prompt)} chars)…", flush=True)
    conditioning = text_encode_qwen_image_edit(
        inputs["clip"],
        prompt,
        inputs.get("vae"),
        image=inputs.get("image"),
    )
    return (conditioning,)


def _text_encode_negative(inputs: dict) -> tuple:
    text = inputs.get("text", "") or ""
    if not text.strip():
        return (fast_empty_negative_conditioning(inputs["clip"]),)
    import nodes

    return nodes.CLIPTextEncode().encode(inputs["clip"], text)


def _empty_sd3_latent(inputs: dict) -> tuple:
    """comfy_extras/nodes_sd3.EmptySD3LatentImage (new API; not in legacy NODE_CLASS_MAPPINGS)."""
    import torch
    import comfy.model_management

    width = int(inputs["width"])
    height = int(inputs["height"])
    batch_size = int(inputs.get("batch_size", 1))
    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return ({"samples": latent, "downscale_ratio_spacial": 8},)


register_node_handler(
    "AnthillQwenImageCLIPLoader",
    _qwen_clip_loader,
    input_types={"required": {"clip_name": ("STRING",)}},
)
register_node_handler(
    "AnthillSetUnionControlNetType",
    _set_union_controlnet_type,
    input_types={
        "required": {
            "control_net": ("CONTROL_NET",),
            "type": ("STRING",),
        }
    },
)
register_node_handler(
    "AnthillTextEncodeQwenImageSource",
    _text_encode_source,
    input_types={
        "required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
        },
        "optional": {
            "vae": ("VAE",),
            "image": ("IMAGE",),
        },
    },
)
register_node_handler(
    "AnthillCLIPTextEncode",
    _text_encode_negative,
    input_types={
        "required": {
            "text": ("STRING", {"multiline": True, "dynamic_prompts": True}),
            "clip": ("CLIP",),
        }
    },
)
register_node_handler(
    "EmptySD3LatentImage",
    _empty_sd3_latent,
    input_types={
        "required": {
            "width": ("INT",),
            "height": ("INT",),
        },
        "optional": {
            "batch_size": ("INT",),
        },
    },
)
