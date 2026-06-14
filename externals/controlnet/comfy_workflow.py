"""Build Comfy workflow for Qwen-Image + InstantX Union ControlNet."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from externals.comfy.client import SEED_MAX, SEED_MIN
from externals.controlnet.model_paths import CLIP_NAME, CONTROLNET_NAME, UNET_NAME, VAE_NAME
from externals.image2image.comfy_workflow import snap_latent_size, stage_input_image

_LATENT_ALIGN = 8
# Comfy Org InstantX ControlNet template (ModelSamplingAuraFlow).
_DEFAULT_SHIFT = 3.1
_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 768
# Soften strength when multiple Union types are chained (official template uses one).
_MULTI_CONTROL_STRENGTH_SCALE = {2: 0.72, 3: 0.52}


def _sampling_shift() -> float:
    raw = os.environ.get("AH_CONTROLNET_SHIFT", "").strip()
    if raw:
        return float(raw)
    return _DEFAULT_SHIFT


def _next_id(counter: list[int]) -> str:
    counter[0] += 1
    return str(counter[0])


def _scale_image_node(
    wf: dict[str, Any],
    counter: list[int],
    *,
    image_ref: str,
    width: int,
    height: int,
) -> str:
    scale_id = _next_id(counter)
    wf[scale_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "image": [image_ref, 0],
            "width": width,
            "height": height,
            "upscale_method": "lanczos",
            "crop": "center",
        },
    }
    return scale_id


def per_control_strength(base_strength: float, control_count: int) -> float:
    """Reduce per-map strength when several Union types are chained."""
    if control_count <= 1:
        return base_strength
    scale = _MULTI_CONTROL_STRENGTH_SCALE.get(control_count, 0.45)
    return round(min(base_strength, base_strength * scale), 3)


def build_controlnet_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    source_image_path: Path | None,
    control_images: list[tuple[str, Path]],
    input_dir: Path,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    denoise: float,
    strength: float,
) -> tuple[dict[str, Any], int]:
    """Return (workflow dict, seed used). control_images: [(union_type, path), ...]."""
    if not control_images:
        raise ValueError("control_images must not be empty")

    wf: dict[str, Any] = {}
    counter = [0]

    unet_id = _next_id(counter)
    wf[unet_id] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": UNET_NAME, "weight_dtype": "fp8_e4m3fn"},
    }

    model_id = _next_id(counter)
    wf[model_id] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": [unet_id, 0], "shift": _sampling_shift()},
    }

    clip_id = _next_id(counter)
    wf[clip_id] = {
        "class_type": "AnthillQwenImageCLIPLoader",
        "inputs": {"clip_name": CLIP_NAME},
    }

    vae_id = _next_id(counter)
    wf[vae_id] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": VAE_NAME},
    }

    cn_loader_id = _next_id(counter)
    wf[cn_loader_id] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": CONTROLNET_NAME},
    }

    if source_image_path is not None:
        staged_source = stage_input_image(
            source_image_path, input_dir, name=f"source_{source_image_path.stem}.png"
        )
        src_load_id = _next_id(counter)
        wf[src_load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": staged_source},
        }
        pixels_id = src_load_id
        if width > 0 and height > 0:
            pixels_id = _scale_image_node(
                wf, counter, image_ref=src_load_id, width=width, height=height
            )
        # denoise < 1: VAE-encode source latent (true img2img). Otherwise keep empty
        # latent and pass the source through vision + reference_latents only.
        use_vae_latent = denoise < 0.99
        if use_vae_latent:
            latent_id = _next_id(counter)
            wf[latent_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [pixels_id, 0], "vae": [vae_id, 0]},
            }
        else:
            latent_id = _next_id(counter)
            wf[latent_id] = {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            }
        pos_id = _next_id(counter)
        wf[pos_id] = {
            "class_type": "AnthillTextEncodeQwenImageSource",
            "inputs": {
                "clip": [clip_id, 0],
                "prompt": prompt,
                "vae": [vae_id, 0],
                "image": [pixels_id, 0],
            },
        }
    else:
        latent_id = _next_id(counter)
        wf[latent_id] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        pos_id = _next_id(counter)
        wf[pos_id] = {
            "class_type": "AnthillCLIPTextEncode",
            "inputs": {"text": prompt, "clip": [clip_id, 0]},
        }

    neg_id = _next_id(counter)
    wf[neg_id] = {
        "class_type": "AnthillCLIPTextEncode",
        "inputs": {"text": negative_prompt, "clip": [clip_id, 0]},
    }

    pos_ref, neg_ref = pos_id, neg_id
    neg_slot = 0
    control_count = len(control_images)
    job_strength = per_control_strength(strength, control_count)
    for index, (union_type, control_path) in enumerate(control_images):
        staged = stage_input_image(
            control_path, input_dir, name=f"ctrl_{index}_{union_type}_{control_path.stem}.png"
        )
        load_id = _next_id(counter)
        wf[load_id] = {"class_type": "LoadImage", "inputs": {"image": staged}}

        ctrl_pixels_id = load_id
        if width > 0 and height > 0:
            ctrl_pixels_id = _scale_image_node(
                wf, counter, image_ref=load_id, width=width, height=height
            )

        type_id = _next_id(counter)
        wf[type_id] = {
            "class_type": "AnthillSetUnionControlNetType",
            "inputs": {"control_net": [cn_loader_id, 0], "type": union_type},
        }

        apply_id = _next_id(counter)
        wf[apply_id] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": [pos_ref, 0],
                "negative": [neg_ref, neg_slot],
                "control_net": [type_id, 0],
                "image": [ctrl_pixels_id, 0],
                "strength": job_strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "vae": [vae_id, 0],
            },
        }
        pos_ref = apply_id
        neg_ref = apply_id
        neg_slot = 1

    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    sampler_id = _next_id(counter)
    wf[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": run_seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise,
            "model": [model_id, 0],
            "positive": [pos_ref, 0],
            "negative": [neg_ref, 1],
            "latent_image": [latent_id, 0],
        },
    }

    decode_id = _next_id(counter)
    wf[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sampler_id, 0], "vae": [vae_id, 0]},
    }

    wf["_output"] = decode_id
    return wf, run_seed


def _cap_size(width: int, height: int) -> tuple[int, int]:
    max_area = int(os.environ.get("AH_CONTROLNET_MAX_AREA", "589824") or "589824")
    out_w, out_h = snap_latent_size(width, height)
    area = out_w * out_h
    if area > max_area and max_area > 0:
        scale = (max_area / area) ** 0.5
        out_w = max(_LATENT_ALIGN, int(out_w * scale) // _LATENT_ALIGN * _LATENT_ALIGN)
        out_h = max(_LATENT_ALIGN, int(out_h * scale) // _LATENT_ALIGN * _LATENT_ALIGN)
    return out_w, out_h


def resolve_job_size(
    *,
    source_path: Path | None,
    control_paths: list[Path],
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    from externals.image2image.comfy_workflow import read_image_size

    if width is not None and height is not None:
        return _cap_size(width, height)

    ref: Path | None = source_path
    if ref is None and control_paths:
        ref = control_paths[0]

    if ref is not None and ref.is_file():
        img_w, img_h = read_image_size(ref)
        out_w = width if width is not None else img_w
        out_h = height if height is not None else img_h
        return _cap_size(out_w, out_h)

    out_w = width if width is not None else _DEFAULT_WIDTH
    out_h = height if height is not None else _DEFAULT_HEIGHT
    return _cap_size(out_w, out_h)
