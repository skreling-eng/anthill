"""$controlnet — Qwen-Image + InstantX Union ControlNet generation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.controlnet.bundle_logic import (
    CONTROL_LABELS,
    ControlCombo,
    passthrough_labels,
    validate_bundle,
)
from externals.controlnet.comfy_runner import get_runner
from externals.controlnet.comfy_workflow import resolve_job_size
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_STEPS = 20
_DEFAULT_CFG = 2.5
_DEFAULT_DENOISE = 1.0
_DEFAULT_DENOISE_VAE_IMG2IMG = 0.55
_DEFAULT_STRENGTH = 0.75
_DEFAULT_STRENGTH_VAE_IMG2IMG = 0.75
_DEFAULT_NEGATIVE = " "

_HELP = """
$controlnet generates images with Qwen-Image + InstantX Union ControlNet.

Bundle layout (from parallel preprocess branches):
  - Control maps (required): images labeled pose, depth, and/or canny
  - Source images (optional): images[] labeled source or img2img
  - Reference @photo without source label → controls only (txt2img), not VAE img2img
  - No source → text-to-image from prompt + control maps only
  - Combos by index: (pose[0], depth[0], canny[0]), …

Example (prompt + controls only):
  @photo: $file('dancer.png')
  @gen: @photo -> (
      $clear -> @photo -> $openpose -> $add_label('pose'),
      $clear -> @photo -> $depth -> $add_label('depth'),
      $clear -> @photo -> $canny -> $add_label('canny'),
  ) -> $controlnet
  studio portrait, dancer pose

Example (optional reference image — style/appearance, not VAE img2img):
  @photo2: $file('dancer2.png')
  @gen: @photo2 -> $add_label('source') -> (
      $clear -> @photo -> $openpose -> $add_label('pose'), …
  ) -> $controlnet

Example (true VAE img2img — lower denoise):
  … -> $controlnet(denoise=0.55, strength=0.75)

Models (FP8, 16GB-friendly with low VRAM):
  models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
  models/qwen-image/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
  models/qwen-image/vae/qwen_image_vae.safetensors
  models/qwen-image/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors

Args:
  prompt=       override prompts[]
  steps=20      sampler steps (Comfy fp8 default: 20)
  cfg=2.5       classifier-free guidance (Comfy fp8 default: 2.5)
  denoise=1.0   default; use denoise<1 with labeled source for VAE img2img (default 0.55)
  strength=0.75  per-controlnet strength (auto-reduced when pose+depth+canny stacked)
  controls=      use subset: pose, depth, canny (comma-separated; default: all present)
  seed=         fixed RNG seed
  width=/height= output size (default: control map or 768×768; source if img2img)
  gpu=1         use CUDA when available

Env:
  AH_CONTROLNET_VRAM=low   default on ≤19GB GPUs
  AH_CONTROLNET_MAX_AREA=589824  (~768×768 cap)
  AH_EMULATE_CONTROLNET=1  stub PNG output
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_CONTROLNET", "").lower() in ("1", "true", "yes")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    return int(raw)


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, str(default)).strip()
    return float(raw)


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _resolve_path(ctx: ExternalContext, link: str) -> Path:
    path = ctx.resolve_link_path(link)
    if not path.is_file():
        raise FileNotFoundError(f"$controlnet: image not found: {path}")
    return path


def _read_prompt(ctx: ExternalContext, inp: ExternalInput) -> str:
    arg_prompt = inp.args.get("prompt", "").strip()
    if arg_prompt:
        return arg_prompt
    texts = read_prompt_texts(ctx, inp)
    if texts:
        return texts[0]
    return ""


def _combo_control_paths(
    ctx: ExternalContext, combo: ControlCombo
) -> list[tuple[str, Path]]:
    return [(union_type, _resolve_path(ctx, link)) for union_type, link in combo.items()]


_CONTROL_ALIASES = {
    "pose": "openpose",
    "openpose": "openpose",
    "depth": "depth",
    "canny": "canny",
}


def _filter_controls(
    controls: list[tuple[str, Path]], controls_arg: str
) -> list[tuple[str, Path]]:
    raw = controls_arg.strip().lower()
    if not raw:
        return controls
    allowed = {
        _CONTROL_ALIASES.get(part.strip(), part.strip())
        for part in raw.split(",")
        if part.strip()
    }
    filtered = [(union_type, path) for union_type, path in controls if union_type in allowed]
    if not filtered:
        raise RuntimeError(
            f"$controlnet: controls={controls_arg!r} matched no maps "
            f"(have: {', '.join(t for t, _ in controls)})"
        )
    return filtered


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.prompts.clear()

    sources, combos = validate_bundle(inp.bundle)
    prompt = _read_prompt(ctx, inp)
    if not prompt.strip():
        raise RuntimeError(_HELP.strip())

    steps = _int_arg(inp.args, "steps", _DEFAULT_STEPS)
    cfg = _float_arg(inp.args, "cfg", _DEFAULT_CFG)
    denoise_arg = inp.args.get("denoise", "").strip()
    strength_arg = inp.args.get("strength", "").strip()
    seed = _optional_int(inp.args, "seed")
    width_arg = _optional_int(inp.args, "width")
    height_arg = _optional_int(inp.args, "height")
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_CONTROLNET_GPU", "1")))
    negative = inp.args.get("neg", inp.args.get("negative_prompt", _DEFAULT_NEGATIVE))
    controls_arg = inp.args.get("controls", "").strip()

    jobs: list[tuple[Path | None, list[tuple[str, Path]], int | None]] = []
    seed_offset = 0
    source_links = sources if sources else [None]
    for source_link in source_links:
        source_path = _resolve_path(ctx, source_link) if source_link else None
        for combo in combos:
            controls = _filter_controls(_combo_control_paths(ctx, combo), controls_arg)
            job_seed = (seed + seed_offset) if seed is not None else None
            seed_offset += 1
            jobs.append((source_path, controls, job_seed))

    mode = "reference" if sources else "txt2img"
    print(
        f"$controlnet: {mode} — {len(source_links)} source slot(s) x {len(combos)} combo(s) "
        f"= {len(jobs)} job(s)",
        flush=True,
    )

    new_images: list[str] = []
    if _emulate_enabled():
        for index, (source_path, controls, job_seed) in enumerate(jobs):
            ctrl = ", ".join(f"{t}" for t, _ in controls)
            source_line = (
                f"source: {source_path.name}\n" if source_path is not None else "source: (prompt only)\n"
            )
            text = (
                f"[emulated $controlnet {index + 1}/{len(jobs)}]\n"
                f"{source_line}"
                f"controls: {ctrl}\n"
                f"prompt: {prompt}\n"
                f"seed: {job_seed}\n"
            )
            new_images.append(ctx.new_link("images", ".png", text.encode("utf-8")))
        out.images = new_images
        out.labels = passthrough_labels(inp.bundle)
        return out

    runner = get_runner(ctx.op_dir, use_gpu=use_gpu)
    t_all = time.perf_counter()
    for index, (source_path, controls, job_seed) in enumerate(jobs):
        control_paths = [path for _, path in controls]
        job_w, job_h = resolve_job_size(
            source_path=source_path,
            control_paths=control_paths,
            width=width_arg,
            height=height_arg,
        )
        if denoise_arg:
            denoise = float(denoise_arg)
        else:
            denoise = _DEFAULT_DENOISE
        if strength_arg:
            strength = float(strength_arg)
        elif source_path is not None and denoise < 0.99:
            strength = _DEFAULT_STRENGTH_VAE_IMG2IMG
        else:
            strength = _DEFAULT_STRENGTH
        from externals.controlnet.comfy_workflow import per_control_strength

        effective_strength = per_control_strength(strength, len(controls))
        job_mode = (
            "vae-img2img"
            if source_path is not None and denoise < 0.99
            else "reference"
            if source_path is not None
            else "txt2img"
        )
        label = source_path.name if source_path is not None else "prompt-only"
        print(
            f"$controlnet: job {index + 1}/{len(jobs)} "
            f"{label} {job_w}x{job_h} mode={job_mode} "
            f"controls={len(controls)} strength={effective_strength} denoise={denoise}",
            flush=True,
        )
        png = runner.run_job(
            prompt=prompt,
            negative_prompt=negative,
            source_image=source_path,
            control_images=controls,
            width=job_w,
            height=job_h,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            strength=strength,
            seed=job_seed,
        )
        new_images.append(ctx.new_link("images", ".png", png))

    print(f"$controlnet: finished {len(jobs)} job(s) in {time.perf_counter() - t_all:.1f}s", flush=True)
    out.images = new_images
    out.labels = passthrough_labels(inp.bundle)
    return out
