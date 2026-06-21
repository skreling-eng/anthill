"""Comfy API workflow for Flux.2 Klein two-reference face swap."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from externals.comfy.client import (
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_IMAGE_ALT,
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
from externals.flux2_klein.comfy_workflow import snap_klein_latent_size, unet_weight_dtype, clip_load_device
from externals.image2image.comfy_workflow import snap_latent_size, stage_input_image

_LATENT_ALIGN = 16
_DEFAULT_CFG = 4.0
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"


def _face_swap_template() -> dict[str, Any]:
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
            "class_type": "LoadImage",
            "inputs": {"image": PLACEHOLDER_IMAGE_ALT},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["4", 0], "vae": ["3", 0]},
        },
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": PLACEHOLDER_PROMPT, "clip": ["2", 0]},
        },
        "9": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["8", 0], "latent": ["6", 0]},
        },
        "10": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["9", 0], "latent": ["7", 0]},
        },
        "11": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]},
        },
        "12": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": 768, "height": 512, "batch_size": 1},
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "seed": PLACEHOLDER_SEED,
                "steps": 20,
                "cfg": _DEFAULT_CFG,
                "sampler_name": _DEFAULT_SAMPLER,
                "scheduler": _DEFAULT_SCHEDULER,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["12", 0],
            },
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["3", 0]},
        },
        "15": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "face_swap", "images": ["14", 0]},
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


def build_face_swap_prompt(
    *,
    prompt: str,
    target_path: Path,
    face_path: Path,
    input_dir: Path,
    model_arg: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    cfg: float,
) -> tuple[dict[str, Any], int]:
    wf = copy.deepcopy(_face_swap_template())
    target_name = stage_input_image(
        target_path, input_dir, name=f"target_{target_path.stem}.png"
    )
    face_name = stage_input_image(
        face_path, input_dir, name=f"face_{face_path.stem}.png"
    )
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_placeholders(
        wf,
        {
            PLACEHOLDER_PROMPT: prompt,
            PLACEHOLDER_IMAGE: target_name,
            PLACEHOLDER_IMAGE_ALT: face_name,
        },
    )
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
