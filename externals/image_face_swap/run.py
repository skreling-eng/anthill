"""$image_face_swap — Flux.2 Klein face/head swap (comfy_lib in-process)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.flux2_klein.model_paths import DEFAULT_MODEL as KLEIN_DEFAULT_MODEL
from externals.image2image.comfy_workflow import resolve_output_size
from externals.image_face_swap.bundle_logic import (
    DEFAULT_PROMPT,
    face_swap_jobs,
    passthrough_labels,
)
from externals.image_face_swap.comfy_runner import (
    run_face_swap,
    run_face_swap_variants,
    save_png_bytes,
)
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_STEPS = 20
_DEFAULT_CFG = 4.0


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE_FACE_SWAP", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return float(raw)


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _repeat_count(inp: ExternalInput) -> int:
    return max(1, inp.repeat) if inp.repeat > 0 else 1


def _resolve_path(ctx: ExternalContext, link: str) -> Path:
    path = Path(link)
    if not path.is_absolute():
        path = (ctx.base_dir / link).resolve()
    return path


def _read_prompt(ctx: ExternalContext, inp: ExternalInput) -> str:
    arg_prompt = inp.args.get("prompt", "").strip()
    if arg_prompt:
        return arg_prompt
    parts = read_prompt_texts(ctx, inp)
    if parts:
        return "\n".join(parts)
    return DEFAULT_PROMPT


def _help() -> str:
    return (
        "$image_face_swap uses Flux.2 Klein 9B FP8 via comfy_lib (in-process).\n"
        "  Target: images[] not labeled 'face'\n"
        "  Donor face: images labeled 'face' ($add_label('face'))\n"
        "  Models: models/flux2klein/flux2Klein9bFp8_fp8.safetensors\n"
        "          + text_encoders/qwen_3_8b_fp8mixed + vae/flux2-vae\n"
        "  Based on: comfy_workflows/Flux 2 Klein Precise Face_Head Swap Final V2.json\n"
        "  Worker venv: AH_EXTERNAL_VENV_image_face_swap=.venvs/media\n"
        "Test without GPU/model: AH_EMULATE_IMAGE_FACE_SWAP=1"
    )


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    prompt: str,
    target: Path,
    face: Path,
) -> None:
    text = (
        f"[emulated $image_face_swap]\n"
        f"prompt: {prompt}\n"
        f"target: {target.name}\n"
        f"face: {face.name}\n"
    )
    out.images.append(ctx.new_link("images", ".png", text.encode("utf-8")))


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.prompts.clear()
    out.images = []

    prompt = _read_prompt(ctx, inp)
    model_arg = inp.args.get("model", KLEIN_DEFAULT_MODEL)
    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _int_arg(inp.args, "steps", _DEFAULT_STEPS)
    cfg = _float_arg(inp.args, "cfg", _DEFAULT_CFG)
    seed = _optional_int(inp.args, "seed")
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_IMAGE_FACE_SWAP_GPU", "1")))
    repeat = _repeat_count(inp)

    try:
        jobs = face_swap_jobs(inp.bundle)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    if _emulate_enabled():
        for job in jobs:
            for _ in range(repeat):
                _emulate(
                    ctx,
                    out,
                    prompt=prompt,
                    target=_resolve_path(ctx, job.target),
                    face=_resolve_path(ctx, job.face),
                )
        out.labels = passthrough_labels(inp.bundle)
        return out

    work_dir = ctx.op_dir / "comfy_work"
    t_load = time.perf_counter()
    print(
        f"$image_face_swap: {len(jobs)} job(s), repeat={repeat}",
        flush=True,
    )

    for job_index, job in enumerate(jobs):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$image_face_swap cancelled")

        target_path = _resolve_path(ctx, job.target)
        face_path = _resolve_path(ctx, job.face)
        job_width, job_height = resolve_output_size(
            target_path, width=width, height=height
        )

        if repeat > 1:
            seeds = [seed + i if seed is not None else None for i in range(repeat)]
            print(
                f"$image_face_swap: job {job_index + 1}/{len(jobs)} "
                f"({job_width}x{job_height}, {repeat} variants)",
                flush=True,
            )
            if job_index == 0:
                print(
                    f"$image_face_swap: pipeline ready in "
                    f"{time.perf_counter() - t_load:.1f}s",
                    flush=True,
                )
            try:
                variants = run_face_swap_variants(
                    work_dir=work_dir,
                    target_path=target_path,
                    face_path=face_path,
                    prompt=prompt,
                    model_arg=model_arg,
                    steps=steps,
                    cfg=cfg,
                    seeds=seeds,
                    width=job_width,
                    height=job_height,
                    use_gpu=use_gpu,
                )
            except ImportError as exc:
                raise RuntimeError(_help()) from exc
            except FileNotFoundError as exc:
                raise RuntimeError(str(exc)) from exc
            for pil in variants:
                out.images.append(ctx.new_link("images", ".png", save_png_bytes(pil)))
            continue

        run_seed = seed + job_index if seed is not None else None
        print(
            f"$image_face_swap: job {job_index + 1}/{len(jobs)} "
            f"({job_width}x{job_height}) target={target_path.name} "
            f"face={face_path.name}",
            flush=True,
        )
        if job_index == 0:
            print(
                f"$image_face_swap: pipeline ready in "
                f"{time.perf_counter() - t_load:.1f}s",
                flush=True,
            )
        try:
            result = run_face_swap(
                work_dir=work_dir,
                target_path=target_path,
                face_path=face_path,
                prompt=prompt,
                model_arg=model_arg,
                steps=steps,
                cfg=cfg,
                seed=run_seed,
                width=job_width,
                height=job_height,
                use_gpu=use_gpu,
            )
        except ImportError as exc:
            raise RuntimeError(_help()) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        out.images.append(ctx.new_link("images", ".png", save_png_bytes(result)))

    out.labels = passthrough_labels(inp.bundle)
    return out
