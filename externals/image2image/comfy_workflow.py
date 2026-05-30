"""Build and patch Qwen-Rapid-AIO Comfy workflow JSON."""

from __future__ import annotations

import copy
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

DEFAULT_WORKFLOW = "Qwen-Rapid-AIO_4.json"
_MAX_ENCODE_IMAGES = 3
_LATENT_ALIGN = 8


def snap_latent_size(width: int, height: int, *, multiple: int = _LATENT_ALIGN) -> tuple[int, int]:
    """Round down to a multiple Comfy latent grids expect."""
    w = max(multiple, int(width) // multiple * multiple)
    h = max(multiple, int(height) // multiple * multiple)
    return w, h


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def resolve_output_size(
    image_path: Path,
    *,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Use explicit width/height when set; otherwise take missing dims from the input image."""
    img_w, img_h = read_image_size(image_path)
    out_w = width if width is not None else img_w
    out_h = height if height is not None else img_h
    return snap_latent_size(out_w, out_h)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_qwen_workflow(ref: str = "") -> dict[str, Any]:
    path = resolve_workflow_path(ref or DEFAULT_WORKFLOW, base_dir=_repo_root())
    return load_workflow_json(path)


def stage_input_image(src: Path, input_dir: Path, *, name: str) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / name
    shutil.copy2(src, dest)
    return name


def _positive_encode_id(prompt: dict[str, Any]) -> str:
    for nid, node in prompt.items():
        if node.get("class_type") != "TextEncodeQwenImageEditPlus":
            continue
        if node.get("inputs", {}).get("prompt", "") != "":
            return nid
    raise KeyError("TextEncodeQwenImageEditPlus (positive) not found in workflow")


def _primary_load_id(prompt: dict[str, Any]) -> str:
    for nid, node in prompt.items():
        if node.get("class_type") == "LoadImage":
            return nid
    raise KeyError("LoadImage not found in workflow")


def build_edit_prompt(
    *,
    prompt: str,
    image_paths: list[Path],
    input_dir: Path,
    checkpoint_name: str,
    seed: int | None,
    width: int,
    height: int,
    steps: int,
    workflow_ref: str = "",
) -> tuple[dict[str, Any], int]:
    """Return (patched API workflow, seed used)."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    wf = copy.deepcopy(load_qwen_workflow(workflow_ref))
    staged: list[str] = []
    for index, path in enumerate(image_paths[:_MAX_ENCODE_IMAGES]):
        staged.append(stage_input_image(path, input_dir, name=f"input_{index}_{path.stem}.png"))

    replacements = {
        PLACEHOLDER_PROMPT: prompt,
        PLACEHOLDER_IMAGE: staged[0],
        PLACEHOLDER_IMAGE_ALT: staged[0],
    }
    wf = patch_placeholders(wf, replacements)
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_seed_placeholder(wf, run_seed)

    for node in wf.values():
        ctype = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if ctype == "CheckpointLoaderSimple":
            inputs["ckpt_name"] = checkpoint_name
        elif ctype == "EmptyLatentImage":
            inputs["width"] = width
            inputs["height"] = height
        elif ctype == "KSampler":
            inputs["steps"] = steps

    load_id = _primary_load_id(wf)
    wf[load_id]["inputs"]["image"] = staged[0]
    encode_id = _positive_encode_id(wf)
    encode_inputs = wf[encode_id]["inputs"]
    encode_inputs.pop("image1", None)
    encode_inputs.pop("image2", None)
    encode_inputs.pop("image3", None)

    for index, name in enumerate(staged):
        key = f"image{index + 1}"
        if index == 0:
            encode_inputs[key] = [load_id, 0]
        else:
            extra_id = f"_load_{index}"
            wf[extra_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            encode_inputs[key] = [extra_id, 0]

    return wf, run_seed
