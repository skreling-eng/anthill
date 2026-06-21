"""Build Comfy API workflows for Flux.2 Klein txt2img and image edit."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

from externals.comfy.client import (
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_PROMPT,
    PLACEHOLDER_SEED,
    SEED_MAX,
    SEED_MIN,
    patch_placeholders,
    patch_seed_placeholder,
)
from externals.flux2_klein.model_paths import (
    TEXT_ENCODER,
    UNET_FILENAME,
    VAE_FILENAME,
    resolve_unet,
)
from externals.image2image.comfy_workflow import snap_latent_size

_LATENT_ALIGN = 16
_DEFAULT_CFG = 1.0
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"


def klein_auto_cap_enabled() -> bool:
    return os.environ.get("AH_FLUX2_KLEIN_AUTO_CAP", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def klein_max_area() -> int | None:
    """Optional pixel-area cap. Default: no cap (respect width/height args).

    Set AH_FLUX2_KLEIN_MAX_AREA to a pixel count, or AH_FLUX2_KLEIN_AUTO_CAP=1
    to shrink large jobs on ≤19 GiB GPUs (edit workflows on 16 GiB cards).
    """
    raw = os.environ.get("AH_FLUX2_KLEIN_MAX_AREA", "").strip()
    if raw:
        return max(_LATENT_ALIGN * _LATENT_ALIGN, int(raw))
    if not klein_auto_cap_enabled():
        return None
    gib = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    if gib <= 19:
        return 589_824  # ~576×1024 — safe for 9B FP8 edit on 16–17 GiB
    if gib <= 24:
        return 768 * 1024
    return 720 * 1280


def snap_klein_latent_size(
    width: int, height: int, *, multiple: int = _LATENT_ALIGN
) -> tuple[int, int]:
    w, h = snap_latent_size(width, height, multiple=multiple)
    max_area = klein_max_area()
    if max_area is None or w * h <= max_area:
        return w, h
    scale = (max_area / (w * h)) ** 0.5
    nw, nh = snap_latent_size(int(w * scale), int(h * scale), multiple=multiple)
    print(
        f"$flux2_klein: capping output {w}x{h} -> {nw}x{nh} "
        f"(max area {max_area}; unset AH_FLUX2_KLEIN_AUTO_CAP / "
        f"AH_FLUX2_KLEIN_MAX_AREA to use requested size)",
        flush=True,
    )
    return nw, nh


def stage_klein_input_image(
    src: Path,
    input_dir: Path,
    *,
    name: str,
    width: int,
    height: int,
) -> str:
    """Stage edit input resized to the output latent size (VAE + reference match KSampler)."""
    from PIL import Image

    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / name
    with Image.open(src) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        img.save(dest, format="PNG")
    return name


def clip_load_device() -> str:
    """Keep Qwen3 8B TE off GPU so the 9B UNet can load for KSampler (~17 GiB cards)."""
    if os.environ.get("AH_FLUX2_KLEIN_CLIP_GPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "default"
    return "cpu"


def unet_weight_dtype(model_arg: str) -> str:
    """FP8 Klein checkpoints need fp8_e4m3fn_fast for comfy_kitchen paths."""
    name = resolve_unet(model_arg).name.lower()
    if "fp8" in name:
        return os.environ.get("AH_FLUX2_KLEIN_UNET_DTYPE", "fp8_e4m3fn_fast")
    return os.environ.get("AH_FLUX2_KLEIN_UNET_DTYPE", "default")


def _txt2img_template() -> dict[str, Any]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_FILENAME,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "flux2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_FILENAME},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": PLACEHOLDER_PROMPT, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": 768, "height": 512, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": PLACEHOLDER_SEED,
                "steps": 20,
                "cfg": _DEFAULT_CFG,
                "sampler_name": _DEFAULT_SAMPLER,
                "scheduler": _DEFAULT_SCHEDULER,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "flux2_klein", "images": ["8", 0]},
        },
    }


def _edit_template() -> dict[str, Any]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNET_FILENAME,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "flux2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_FILENAME},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": PLACEHOLDER_IMAGE},
        },
        "5": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["4", 0], "vae": ["3", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": PLACEHOLDER_PROMPT, "clip": ["2", 0]},
        },
        "7": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["6", 0], "latent": ["5", 0]},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]},
        },
        "9": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": 768, "height": 512, "batch_size": 1},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": PLACEHOLDER_SEED,
                "steps": 20,
                "cfg": _DEFAULT_CFG,
                "sampler_name": _DEFAULT_SAMPLER,
                "scheduler": _DEFAULT_SCHEDULER,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "flux2_klein_edit", "images": ["11", 0]},
        },
    }


def _patch_clip_device(wf: dict[str, Any]) -> None:
    device = clip_load_device()
    for node in wf.values():
        if node.get("class_type") == "CLIPLoader":
            node.setdefault("inputs", {})["device"] = device


def _patch_unet_name(wf: dict[str, Any], model_arg: str) -> None:
    unet_name = resolve_unet(model_arg).name
    weight_dtype = unet_weight_dtype(model_arg)
    for node in wf.values():
        if node.get("class_type") == "UNETLoader":
            inputs = node.setdefault("inputs", {})
            inputs["unet_name"] = unet_name
            inputs["weight_dtype"] = weight_dtype


def build_txt2img_prompt(
    *,
    prompt: str,
    model_arg: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    cfg: float,
) -> tuple[dict[str, Any], int]:
    wf = copy.deepcopy(_txt2img_template())
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_placeholders(wf, {PLACEHOLDER_PROMPT: prompt})
    wf = patch_seed_placeholder(wf, run_seed)
    _patch_unet_name(wf, model_arg)
    _patch_clip_device(wf)
    out_w, out_h = snap_klein_latent_size(width, height, multiple=_LATENT_ALIGN)
    for node in wf.values():
        ctype = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if ctype == "EmptyFlux2LatentImage":
            inputs["width"] = out_w
            inputs["height"] = out_h
        elif ctype == "KSampler":
            inputs["steps"] = steps
            inputs["cfg"] = cfg
    return wf, run_seed


def build_edit_prompt(
    *,
    prompt: str,
    image_path: Path,
    input_dir: Path,
    model_arg: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    cfg: float,
) -> tuple[dict[str, Any], int]:
    wf = copy.deepcopy(_edit_template())
    out_w, out_h = snap_klein_latent_size(width, height, multiple=_LATENT_ALIGN)
    staged = stage_klein_input_image(
        image_path,
        input_dir,
        name=f"klein_input_{image_path.stem}.png",
        width=out_w,
        height=out_h,
    )
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_placeholders(wf, {PLACEHOLDER_PROMPT: prompt, PLACEHOLDER_IMAGE: staged})
    wf = patch_seed_placeholder(wf, run_seed)
    _patch_unet_name(wf, model_arg)
    _patch_clip_device(wf)
    for node in wf.values():
        ctype = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if ctype == "EmptyFlux2LatentImage":
            inputs["width"] = out_w
            inputs["height"] = out_h
        elif ctype == "KSampler":
            inputs["steps"] = steps
            inputs["cfg"] = cfg
    return wf, run_seed
