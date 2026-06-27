"""Build Comfy API workflow for SkyReels V3 talking avatars."""

from __future__ import annotations

import copy
import os
import random
import shutil
from pathlib import Path
from typing import Any

from externals.avatar.model_paths import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_TEXT_ENCODER,
    DEFAULT_VAE,
    DEFAULT_WAN_MODEL,
    DEFAULT_WAV2VEC,
    DEFAULT_WORKFLOW,
    default_attention_mode,
    fit_avatar_resolution,
    resolve_blocks_to_swap,
    resolve_force_offload,
    resolve_tiled_vae,
    resolve_wan_load_device,
)
from externals.comfy.client import (
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_PROMPT,
    PLACEHOLDER_SEED,
    PLACEHOLDER_SOUND,
    SEED_MAX,
    SEED_MIN,
    load_workflow_json,
    patch_placeholders,
    patch_seed_placeholder,
    resolve_workflow_path,
)
from externals.image2image.comfy_workflow import read_image_size, snap_latent_size

PLACEHOLDER_NEG_PROMPT = "INPUT_NEG_PROMPT"
_LATENT_ALIGN = 16
_REPO_ROOT = Path(__file__).resolve().parents[2]


def stage_input_file(src: Path, input_dir: Path, *, name: str) -> str:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / name
    shutil.copy2(src, dest)
    return name


def resolve_avatar_size(
    image_path: Path,
    *,
    width: int | None,
    height: int | None,
    cap_unspecified: bool = True,
) -> tuple[int, int]:
    img_w, img_h = read_image_size(image_path)
    user_set = width is not None or height is not None
    out_w = width if width is not None else img_w
    out_h = height if height is not None else img_h
    out_w, out_h = snap_latent_size(out_w, out_h, multiple=_LATENT_ALIGN)
    if cap_unspecified and not user_set:
        capped_w, capped_h = fit_avatar_resolution(out_w, out_h)
        if capped_w != out_w or capped_h != out_h:
            print(
                f"$avatar: output size {out_w}x{out_h} -> {capped_w}x{capped_h} "
                f"(AVATAR_MAX_AREA cap; unset it to keep full input resolution)",
                flush=True,
            )
        out_w, out_h = capped_w, capped_h
    return out_w, out_h


def _load_template(ref: str = "") -> dict[str, Any]:
    path = resolve_workflow_path(ref or DEFAULT_WORKFLOW, base_dir=_REPO_ROOT)
    return load_workflow_json(path)


def _patch_avatar_nodes(
    wf: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    fps: float,
    num_frames: int,
    frame_window_size: int,
    motion_frame: int,
    drop_frames: int,
    wan_model: str,
    vae_name: str,
    text_encoder: str,
    wav2vec_model: str,
    cfg: float,
    blocks_to_swap: int | None = None,
    tiled_vae: bool | None = None,
) -> None:
    swap = resolve_blocks_to_swap(blocks_to_swap)
    load_device = resolve_wan_load_device(swap)
    force_offload = resolve_force_offload(swap)
    use_tiled_vae = resolve_tiled_vae(tiled_vae)
    for node in wf.values():
        ctype = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if ctype == "WanVideoBlockSwap":
            inputs["blocks_to_swap"] = swap
        elif ctype == "WanVideoModelLoader":
            inputs["model"] = wan_model
            inputs["attention_mode"] = default_attention_mode()
            inputs["load_device"] = load_device
            if swap == 0:
                inputs.pop("block_swap_args", None)
        elif ctype == "WanVideoVAELoader":
            inputs["model_name"] = vae_name
        elif ctype == "WanVideoTextEncodeCached":
            inputs["model_name"] = text_encoder
            inputs["positive_prompt"] = prompt
            inputs["negative_prompt"] = negative_prompt
        elif ctype == "DownloadAndLoadWav2VecModel":
            inputs["model"] = wav2vec_model
        elif ctype == "MultiTalkWav2VecEmbeds":
            inputs["num_frames"] = num_frames
            inputs["fps"] = fps
        elif ctype == "WanVideoImageToVideoSkyreelsv3_audio":
            inputs["width"] = width
            inputs["height"] = height
            inputs["frame_window_size"] = frame_window_size
            inputs["motion_frame"] = motion_frame
            inputs["drop_frames"] = drop_frames
            inputs["force_offload"] = force_offload
            inputs["tiled_vae"] = use_tiled_vae
        elif ctype == "WanVideoSchedulerv2":
            inputs["steps"] = steps
        elif ctype == "WanVideoSamplerv2":
            inputs["cfg"] = cfg
            inputs["force_offload"] = force_offload


def build_avatar_prompt(
    *,
    prompt: str,
    negative_prompt: str,
    image_path: Path,
    audio_path: Path,
    input_dir: Path,
    seed: int | None,
    width: int,
    height: int,
    steps: int = 4,
    fps: float = 24.0,
    num_frames: int = 400,
    frame_window_size: int = 81,
    motion_frame: int = 5,
    drop_frames: int = 12,
    wan_model: str = DEFAULT_WAN_MODEL,
    vae_name: str = DEFAULT_VAE,
    text_encoder: str = DEFAULT_TEXT_ENCODER,
    wav2vec_model: str = DEFAULT_WAV2VEC,
    cfg: float = 1.0,
    workflow_ref: str = "",
    blocks_to_swap: int | None = None,
    tiled_vae: bool | None = None,
) -> tuple[dict[str, Any], int]:
    """Return (patched API workflow, seed used)."""
    wf = copy.deepcopy(_load_template(workflow_ref))
    image_name = stage_input_file(
        image_path, input_dir, name=f"avatar_{image_path.stem}.png"
    )
    audio_name = stage_input_file(
        audio_path, input_dir, name=f"avatar_{audio_path.stem}{audio_path.suffix}"
    )

    replacements = {
        PLACEHOLDER_PROMPT: prompt,
        PLACEHOLDER_NEG_PROMPT: negative_prompt,
        PLACEHOLDER_IMAGE: image_name,
        PLACEHOLDER_SOUND: audio_name,
    }
    wf = patch_placeholders(wf, replacements)
    run_seed = seed if seed is not None else random.randint(SEED_MIN, SEED_MAX)
    wf = patch_seed_placeholder(wf, run_seed)

    _patch_avatar_nodes(
        wf,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        fps=fps,
        num_frames=num_frames,
        frame_window_size=frame_window_size,
        motion_frame=motion_frame,
        drop_frames=drop_frames,
        wan_model=wan_model,
        vae_name=vae_name,
        text_encoder=text_encoder,
        wav2vec_model=wav2vec_model,
        cfg=cfg,
        blocks_to_swap=blocks_to_swap,
        tiled_vae=tiled_vae,
    )

    for nid, node in wf.items():
        if node.get("class_type") != "LoadImage":
            continue
        node["inputs"]["image"] = image_name
    for nid, node in wf.items():
        if node.get("class_type") != "LoadAudio":
            continue
        node["inputs"]["audio"] = audio_name

    return wf, run_seed
