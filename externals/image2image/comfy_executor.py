"""Execute Comfy API-format workflow JSON using comfy_lib nodes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from externals.comfy_inprocess.executor import (  # noqa: F401
    ComfyWorkflowError,
    execute_prompt,
    execute_prompt_legacy,
    find_node_id,
    register_node_handler,
    topo_order as _topo_order,
)
from externals.image2image.comfy_qwen_nodes import (
    fast_empty_negative_conditioning,
    text_encode_qwen_image_edit_plus,
)


def _qwen_encode_handler(inputs: dict) -> tuple:
    prompt = inputs.get("prompt", "") or ""
    if not prompt.strip() and all(inputs.get(f"image{i}") is None for i in (1, 2, 3)):
        return (fast_empty_negative_conditioning(inputs["clip"]),)
    if prompt.strip():
        print(
            f"$image2image: encoding prompt ({len(prompt)} chars, vision+text)…",
            flush=True,
        )
    t0 = time.perf_counter()
    result = (
        text_encode_qwen_image_edit_plus(
            inputs["clip"],
            inputs.get("prompt", ""),
            inputs.get("vae"),
            image1=inputs.get("image1"),
            image2=inputs.get("image2"),
            image3=inputs.get("image3"),
        ),
    )
    if prompt.strip():
        print(
            f"$image2image: encode done ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    return result


_QWEN_EDIT_PLUS_INPUT_TYPES = {
    "required": {
        "clip": ("CLIP",),
        "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
    },
    "optional": {
        "vae": ("VAE",),
        "image1": ("IMAGE",),
        "image2": ("IMAGE",),
        "image3": ("IMAGE",),
    },
}

register_node_handler(
    "TextEncodeQwenImageEditPlus",
    _qwen_encode_handler,
    input_types=_QWEN_EDIT_PLUS_INPUT_TYPES,
)

_CHECKPOINT_CACHE: dict[str, tuple[Any, ...]] = {}


def _cached_checkpoint_loader(inputs: dict) -> tuple[Any, ...]:
    """Reuse loaded MODEL/CLIP/VAE in warm worker (avoid re-reading multi-GB safetensors)."""
    import folder_paths
    import nodes

    ckpt_name = inputs["ckpt_name"]
    ckpt_path = Path(folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)).resolve()
    key = str(ckpt_path)
    if key not in _CHECKPOINT_CACHE:
        print(
            f"$image2image: reading checkpoint from disk ({ckpt_name})…",
            flush=True,
        )
        t0 = time.perf_counter()
        cls = nodes.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]
        out = cls().load_checkpoint(ckpt_name)
        _CHECKPOINT_CACHE[key] = out
        from externals.image2image.comfy_sample_prep import reset_unet_prepared

        reset_unet_prepared()
        print(
            f"$image2image: loaded checkpoint {ckpt_name} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    else:
        print(f"$image2image: checkpoint cache hit ({ckpt_name})", flush=True)
    out = _CHECKPOINT_CACHE[key]
    from externals.image2image.comfy_sample_prep import preload_for_encode

    t_gpu = time.perf_counter()
    preload_for_encode(out)
    print(
        f"$image2image: CLIP/VAE on GPU ({time.perf_counter() - t_gpu:.1f}s)",
        flush=True,
    )
    return out


register_node_handler(
    "CheckpointLoaderSimple",
    _cached_checkpoint_loader,
    input_types={"required": {"ckpt_name": ("STRING",)}},
)
