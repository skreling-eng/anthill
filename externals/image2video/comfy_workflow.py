"""Build and patch Wan I2V Comfy workflow JSON (API /prompt format)."""

from __future__ import annotations

import copy
import os
import random
import shutil
from pathlib import Path
from typing import Any

from externals.comfy.client import (
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_IMAGE_ALT,
    PLACEHOLDER_PROMPT,
    PLACEHOLDER_SEED,
    SEED_MAX,
    SEED_MIN,
    load_workflow_json,
    patch_placeholders,
    patch_seed_placeholder,
    resolve_workflow_path,
)
from externals.image2image.comfy_workflow import (
    read_image_size,
    snap_latent_size,
)
from externals.image2video.model_paths import (
    DEFAULT_CLIP_VISION,
    MEGA_WORKFLOW,
    comfy_ckpt_name,
    is_mega_model,
)

DEFAULT_WORKFLOW = "Wan_i2v_rapid__start_image.json"
_LATENT_ALIGN = 16


def resolve_output_size(
    image_path: Path,
    *,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    img_w, img_h = read_image_size(image_path)
    out_w = width if width is not None else img_w
    out_h = height if height is not None else img_h
    return snap_latent_size(out_w, out_h, multiple=_LATENT_ALIGN)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_i2v_workflow(ref: str = "") -> dict[str, Any]:
    path = resolve_workflow_path(ref or DEFAULT_WORKFLOW, base_dir=_repo_root())
    return load_workflow_json(path)


def stage_input_image(src: Path, input_dir: Path, *, name: str) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / name
    shutil.copy2(src, dest)
    return name


def build_i2v_prompt(
    *,
    prompt: str,
    image_path: Path,
    input_dir: Path,
    checkpoint_name: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    num_frames: int,
    negative_prompt: str = "",
    guidance: float | None = None,
    workflow_ref: str = "",
) -> tuple[dict[str, Any], int]:
    """Return (patched API workflow, seed used)."""
    wf = copy.deepcopy(load_i2v_workflow(workflow_ref))
    staged = stage_input_image(
        image_path, input_dir, name=f"start_{image_path.stem}.png"
    )

    replacements = {
        PLACEHOLDER_PROMPT: prompt,
        PLACEHOLDER_IMAGE: staged,
        PLACEHOLDER_IMAGE_ALT: staged,
        "IMAGE_INPUT": staged,
        "INPUT_IMAGE": staged,
    }
    wf = patch_placeholders(wf, replacements)
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_seed_placeholder(wf, run_seed)

    for node in wf.values():
        ctype = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if ctype == "CheckpointLoaderSimple":
            inputs["ckpt_name"] = checkpoint_name
        elif ctype in ("WanImageToVideo", "WanVaceToVideo"):
            inputs["width"] = width
            inputs["height"] = height
            inputs["length"] = num_frames
            if ctype == "WanVaceToVideo":
                inputs["strength"] = 1
        elif ctype == "CLIPVisionLoader":
            clip_name = os.environ.get("WAN_I2V_CLIP_VISION", "").strip()
            if clip_name:
                inputs["clip_name"] = clip_name
            else:
                inputs["clip_name"] = DEFAULT_CLIP_VISION
        elif ctype == "KSampler":
            inputs["steps"] = steps
            if guidance is not None:
                inputs["cfg"] = guidance
        elif ctype == "CLIPTextEncode":
            title = (node.get("_meta") or {}).get("title", "").lower()
            if "negative" in title or "neg" in title:
                inputs["text"] = negative_prompt
            elif "prompt" in title or ctype == "CLIPTextEncode":
                if "negative" not in title:
                    inputs["text"] = prompt
        elif ctype == "PrimitiveInt" and (node.get("_meta") or {}).get("title") == "Number of Frames":
            inputs["value"] = num_frames

    for nid, node in wf.items():
        if node.get("class_type") != "LoadImage":
            continue
        meta = (node.get("_meta") or {}).get("title", "")
        if "End" in meta and "Start" not in meta:
            continue
        node["inputs"]["image"] = staged

    return wf, run_seed


def build_i2v_prompt_for_model(
    *,
    prompt: str,
    image_path: Path,
    input_dir: Path,
    model_arg: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    num_frames: int,
    negative_prompt: str = "",
    guidance: float | None = None,
    workflow_ref: str = "",
) -> tuple[dict[str, Any], int]:
    wf_ref = workflow_ref
    if not wf_ref and is_mega_model(model_arg):
        wf_ref = MEGA_WORKFLOW
    return build_i2v_prompt(
        prompt=prompt,
        image_path=image_path,
        input_dir=input_dir,
        checkpoint_name=comfy_ckpt_name(model_arg),
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        num_frames=num_frames,
        negative_prompt=negative_prompt,
        guidance=guidance,
        workflow_ref=wf_ref,
    )
