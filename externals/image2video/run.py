"""$image2video — animate images into video (Wan MEGA I2V, start frame = input image)."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list, read_prompt_texts
from externals.image2video.comfy_workflow import resolve_output_size
from externals.image2video.model_list import DEFAULT_NEGATIVE_PROMPT
from ahlib.ah_runtime import ArrayBundle


def _output_name(model: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", model)
    return f"{ts}_{safe}_i2v_{index}.mp4"


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE2VIDEO", "").lower() in (
        "1",
        "true",
        "yes",
    )


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


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_text(ctx: ExternalContext, inp: ExternalInput) -> str:
    if inp.args.get("prompt", "").strip():
        return inp.args["prompt"].strip()
    parts = read_prompt_texts(ctx, inp)
    if parts:
        return "\n".join(parts)
    return inp.prompt_text.strip()


def _image_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.images:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _path_to_link(ctx: ExternalContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ctx.base_dir.resolve())).replace("\\", "/")
    except ValueError:
        pass
    dest = ctx.op_dir / "videos" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return str(dest.relative_to(ctx.base_dir)).replace("\\", "/")


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    model: str,
    prompt: str,
    images: list[Path],
) -> None:
    for image_path in images:
        content = (
            f"[emulated $image2video model={model}]\n"
            f"prompt: {prompt}\n"
            f"start_image: {image_path.name}\n"
        )
        out.videos.append(ctx.new_link("videos", ".mp4", content.encode("utf-8")))
    if not images and prompt:
        out.videos.append(
            ctx.new_link(
                "videos",
                ".mp4",
                f"[emulated $image2video]\n{prompt}\n".encode("utf-8"),
            )
        )


def _resolve_backend(inp: ExternalInput) -> str:
    raw = (
        inp.args.get("backend", "").strip()
        or os.environ.get("AH_IMAGE2VIDEO_BACKEND", "").strip()
        or "comfy_lib"
    ).lower()
    if raw in ("comfy", "comfyui"):
        return "comfy"
    if raw in ("diffusers", "wan"):
        return "diffusers"
    return "comfy_lib"


def _run_comfy_lib(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    *,
    models: list[str],
    prompt: str,
    image_paths: list[Path],
) -> None:
    from externals.comfy_inprocess.vae_tiling import configure_tiled_vae_for_job
    from externals.comfy_inprocess.vram_config import configure_comfy_vram_for_job
    from externals.image2video.comfy_runner import run_comfy_i2v
    from externals.image2video.model_list import get_video_model

    configure_tiled_vae_for_job(inp.args)
    configure_comfy_vram_for_job(inp.args)

    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _optional_int(inp.args, "steps")
    guidance = _optional_float(inp.args, "guidance", "cfg")
    frames = _optional_int(inp.args, "frames")
    seed = _optional_int(inp.args, "seed") or 0
    neg = inp.args.get("neg", inp.args.get("negative_prompt", ""))

    work_dir = ctx.op_dir / "comfy_work"
    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    track = 0

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    for model_name in models:
        from externals.comfy_inprocess.memory_guard import configure_mega_runtime_defaults
        from externals.comfy_inprocess.vram_config import apply_comfy_vram_settings
        from externals.image2video.model_paths import is_mega_model

        configure_mega_runtime_defaults(model_name, inp.args)
        apply_comfy_vram_settings()
        profile = get_video_model(model_name)
        job_steps = steps if steps is not None else profile.num_inference_steps
        job_guidance = guidance if guidance is not None else profile.guidance_scale
        job_frames = frames if frames is not None else profile.num_frames
        raw_frames = os.environ.get("WAN_I2V_FRAMES", "").strip()
        if raw_frames:
            job_frames = int(raw_frames)

        for image_path in image_paths:
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                from ahlib.ah_runtime import RuntimeCancelled

                raise RuntimeCancelled("$image2video cancelled")
            job_width, job_height = resolve_output_size(
                image_path, width=width, height=height
            )
            from externals.comfy_inprocess.memory_guard import apply_wan_memory_limits

            req_w, req_h, req_f = job_width, job_height, job_frames
            job_width, job_height, job_frames = apply_wan_memory_limits(
                width=job_width,
                height=job_height,
                num_frames=job_frames,
                mega=is_mega_model(model_name),
            )
            if (req_w, req_h, req_f) != (job_width, job_height, job_frames):
                print(
                    f"$image2video: VRAM cap applied "
                    f"(requested {req_w}x{req_h}, {req_f} frames → "
                    f"{job_width}x{job_height}, {job_frames} frames; "
                    "WAN_I2V_AUTO_CAP=0 to disable).",
                    flush=True,
                )
            out_path = videos_dir / _output_name(model_name, track)
            run_seed = seed + track if seed else seed
            extras: list[str] = []
            if os.environ.get("WAN_I2V_TILED_VAE", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                extras.append("tiled_vae=1")
            vram_mode = os.environ.get("WAN_I2V_VRAM", "").strip().lower()
            if vram_mode and vram_mode not in ("normal", "default", "off", "0"):
                extras.append(f"vram={vram_mode}")
            extra_note = (" " + " ".join(extras)) if extras else ""
            print(
                f"$image2video: I2V {track + 1} model={model_name} "
                f"{job_width}x{job_height} frames={job_frames} steps={job_steps}{extra_note}",
                flush=True,
            )
            workflow_ref = (
                inp.args.get("workflow", "").strip()
                or inp.args.get("json", "").strip()
            )
            run_comfy_i2v(
                work_dir=work_dir,
                image_path=image_path,
                prompt=prompt,
                output_path=out_path,
                model_arg=model_name,
                steps=job_steps,
                seed=run_seed,
                width=job_width,
                height=job_height,
                num_frames=job_frames,
                negative_prompt=neg or DEFAULT_NEGATIVE_PROMPT,
                guidance=job_guidance,
                fps=profile.fps,
                workflow_ref=workflow_ref,
            )
            out.videos.append(_path_to_link(ctx, out_path))
            track += 1


def _run_diffusers(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    *,
    models: list[str],
    prompt: str,
    image_paths: list[Path],
) -> None:
    import gc

    import torch
    from externals.image2video import wan_i2v
    from externals.image2video.model_list import get_video_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _optional_int(inp.args, "steps")
    guidance = _optional_float(inp.args, "guidance", "cfg")
    frames = _optional_int(inp.args, "frames")
    seed = _int_arg(inp.args, "seed", 0)
    neg = inp.args.get("neg", inp.args.get("negative_prompt", ""))
    attn = (
        inp.args.get("attn", "").strip()
        or inp.args.get("attention", "").strip()
        or "sage"
    )

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    track = 0

    for model_name in models:
        video_model = get_video_model(model_name)
        for image_path in image_paths:
            out_path = videos_dir / _output_name(model_name, track)
            run_seed = seed + track if seed else seed
            wan_i2v.generate(
                video_model,
                image_path=image_path,
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


def _run_comfy_api(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    *,
    models: list[str],
    prompt: str,
    image_paths: list[Path],
) -> None:
    from externals.image2video.comfy_api import generate_via_comfy
    from externals.image2video.model_list import get_video_model

    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _optional_int(inp.args, "steps")
    guidance = _optional_float(inp.args, "guidance", "cfg")
    seed = _int_arg(inp.args, "seed", 0)
    neg = inp.args.get("neg", inp.args.get("negative_prompt", ""))

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    track = 0

    for model_name in models:
        profile = get_video_model(model_name)
        for image_path in image_paths:
            out_path = videos_dir / _output_name(model_name, track)
            run_seed = seed + track if seed else seed
            generate_via_comfy(
                image_path=image_path,
                prompt=prompt,
                output_path=out_path,
                negative_prompt=neg or DEFAULT_NEGATIVE_PROMPT,
                seed=run_seed,
                steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            out.videos.append(_path_to_link(ctx, out_path))
            track += 1
        _ = profile


def _help() -> str:
    from externals.image2video.model_paths import available_models

    return (
        "$image2video uses comfy_lib (in-process) with Wan_i2v_rapid__start_image.json.\n"
        "  model= — default mega (wan2.2-rapid-mega-aio-v12); rapid: model=wan or model=rapid.\n"
        "  model=mega-nsfw — NSFW MEGA checkpoint (v12.2).\n"
        "  Worker venv: .venvs/comfy-wan (tools/setup_external_venvs.ps1 or init.bat).\n"
        "  Re-sync after pyproject changes: "
        "UV_PROJECT_ENVIRONMENT=.venvs/comfy-wan uv sync --extra media,comfy-wan,clip\n"
        "  Optional ComfyUI python: AH_COMFY_PYTHON (comfy_aimdo / comfy_kitchen).\n"
        "  Checkpoint: models/wan/ or ComfyUI models/checkpoints/WAN/ "
        f"({available_models()})\n"
        "  backend=diffusers — diffusers Wan path\n"
        "  backend=comfy — ComfyUI HTTP API (COMFYUI_WORKFLOW)\n"
        "  width/height/frames from $image2video(...) are used as-is (WAN_I2V_AUTO_CAP off by default).\n"
        "  Optional WAN_I2V_AUTO_CAP=1 shrinks resolution/frames on ≤18GB GPUs.\n"
        "  16GB MEGA: WAN_I2V_VRAM=novram still auto-set unless overridden; use vram=novram if OOM.\n"
        "Test without GPU/model: AH_EMULATE_IMAGE2VIDEO=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.prompts.clear()
    models = read_arg_list(inp, "model", "mega")
    prompt = _prompt_text(ctx, inp)
    image_paths = _image_paths(ctx, inp.bundle)
    backend = _resolve_backend(inp)

    if _emulate_enabled():
        _emulate(ctx, out, model=models[0], prompt=prompt, images=image_paths)
        out.images.clear()
        return out

    if not image_paths:
        if prompt:
            _emulate(ctx, out, model=models[0], prompt=prompt, images=[])
        out.images.clear()
        return out

    if not prompt:
        raise ValueError("$image2video requires a prompt (instruction body or prompts[])")

    try:
        if backend == "comfy_lib":
            _run_comfy_lib(
                ctx, inp, out, models=models, prompt=prompt, image_paths=image_paths
            )
        elif backend == "comfy":
            _run_comfy_api(
                ctx, inp, out, models=models, prompt=prompt, image_paths=image_paths
            )
        else:
            _run_diffusers(
                ctx, inp, out, models=models, prompt=prompt, image_paths=image_paths
            )
    except ImportError as exc:
        raise RuntimeError(_help()) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    out.images.clear()
    return out
