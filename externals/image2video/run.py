"""$image2video — animate images into video (Wan I2V, start frame = input image)."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list, read_prompt_texts
from ahlib.ah_runtime import ArrayBundle


def _output_name(model: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", model)
    return f"{ts}_{safe}_i2v_{index}.mp4"


def _prompt_text(ctx: ExternalContext, inp: ExternalInput) -> str:
    parts = read_prompt_texts(ctx, inp)
    if parts:
        return "\n".join(parts)
    return inp.prompt_text.strip()


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    model: str,
    prompt: str,
    images: list[str],
) -> ArrayBundle:
    for i, img_link in enumerate(images):
        content = (
            f"[emulated $image2video model={model}]\n"
            f"prompt: {prompt}\n"
            f"start_image: {img_link}\n"
        )
        link = ctx.new_link("videos", ".mp4", content)
        out.videos.append(link)
    if not images and prompt:
        link = ctx.new_link("videos", ".mp4", f"[emulated $image2video]\n{prompt}\n")
        out.videos.append(link)
    return out


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _optional_float(args: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        raw = args.get(key, "").strip()
        if raw:
            return float(raw)
    return None


def _path_to_link(ctx: ExternalContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ctx.base_dir.resolve())).replace("\\", "/")
    except ValueError:
        pass
    dest = ctx.op_dir / "videos" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return str(dest.relative_to(ctx.base_dir)).replace("\\", "/")


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    models = read_arg_list(inp, "model", "wan")
    prompt = _prompt_text(ctx, inp)
    images = list(inp.bundle.images)
    seed = int(inp.args.get("seed", "0"))
    neg = inp.args.get("neg", inp.args.get("negative_prompt", ""))
    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _optional_int(inp.args, "steps")
    guidance = _optional_float(inp.args, "guidance", "cfg")
    frames = _optional_int(inp.args, "frames")
    attn = (
        inp.args.get("attn", "").strip()
        or inp.args.get("attention", "").strip()
        or "sage"
    )
    backend = (
        inp.args.get("backend", "").strip()
        or os.environ.get("AH_IMAGE2VIDEO_BACKEND", "").strip()
        or "diffusers"
    ).lower()

    if os.environ.get("AH_EMULATE_IMAGE2VIDEO", "").lower() in ("1", "true", "yes"):
        _emulate(ctx, out, model=models[0], prompt=prompt, images=images)
        out.images.clear()
        return out

    if not images:
        if prompt:
            _emulate(ctx, out, model=models[0], prompt=prompt, images=[])
        out.images.clear()
        return out

    if not prompt:
        raise ValueError("$image2video requires a prompt (instruction body or prompts[])")

    try:
        from externals.image2video.model_list import DEFAULT_NEGATIVE_PROMPT

        videos_dir = ctx.op_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        track = 0

        use_comfy = backend in ("comfy", "comfyui")

        if use_comfy:
            from externals.image2video.comfy_api import generate_via_comfy
        else:
            import gc

            import torch
            from externals.image2video.model_list import get_video_model
            from externals.image2video import wan_i2v

            # Same run as $image: free Flux weights before loading Wan (~20GB VRAM).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for model_name in models:
            video_model = None if use_comfy else get_video_model(model_name)
            for img_link in images:
                src = (ctx.base_dir / img_link).resolve()
                if not src.is_file():
                    raise FileNotFoundError(f"$image2video: image not found: {src}")

                out_path = videos_dir / _output_name(model_name, track)
                run_seed = seed + track if seed else seed
                if use_comfy:
                    generate_via_comfy(
                        image_path=src,
                        prompt=prompt,
                        output_path=out_path,
                        negative_prompt=neg or DEFAULT_NEGATIVE_PROMPT,
                        seed=run_seed,
                        steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                    )
                else:
                    wan_i2v.generate(
                        video_model,
                        image_path=src,
                        prompt=prompt,
                        output_path=out_path,
                        negative_prompt=neg or DEFAULT_NEGATIVE_PROMPT,
                        seed=run_seed,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        num_frames=frames,
                        attn=attn,
                    )
                out.videos.append(_path_to_link(ctx, out_path))
                track += 1

    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"$image2video fallback to emulate ({exc})")
        _emulate(ctx, out, model=models[0], prompt=prompt, images=images)

    out.images.clear()
    return out
