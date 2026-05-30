"""$image2image — Qwen-Rapid-AIO prompt-guided image edit (comfy_lib in-process)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.image2image.comfy_workflow import resolve_output_size
from externals.image2image.comfy_runner import run_comfy_edit, save_png_bytes
from externals.image2image.model_paths import DEFAULT_MODEL, available_models
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_STEPS = 4
_MAX_USE_ALL_IMAGES = 4
_MAX_COMFY_ENCODE_IMAGES = 3


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE2IMAGE", "").lower() in (
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


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _image_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.images:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _prompts_for_images(ctx: ExternalContext, inp: ExternalInput, count: int) -> list[str]:
    if inp.args.get("prompt", "").strip():
        return [inp.args["prompt"].strip()] * count
    prompts = read_prompt_texts(ctx, inp)
    if not prompts:
        return [""] * count
    if len(prompts) == 1:
        return [prompts[0]] * count
    if len(prompts) >= count:
        return prompts[:count]
    padded = list(prompts)
    padded.extend([prompts[-1]] * (count - len(prompts)))
    return padded


def _help() -> str:
    return (
        "$image2image uses comfy_lib (in-process ComfyUI) with Qwen-Rapid-AIO_4.json.\n"
        "  Set AH_COMFY_PYTHON to ComfyUI venv python (needs comfy_aimdo).\n"
        "  Default tries G:\\ComfyUI_V\\.venv\\Scripts\\python.exe\n"
        "  Checkpoint: models/qwen-rapid/ "
        f"({available_models()})\n"
        "Test without GPU/model: AH_EMULATE_IMAGE2IMAGE=1"
    )


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    prompt: str,
    image_paths: list[Path],
    use_all: bool,
) -> None:
    if use_all:
        names = ", ".join(p.name for p in image_paths[:_MAX_USE_ALL_IMAGES])
        text = (
            f"[emulated $image2image use_all=True]\n"
            f"prompt: {prompt}\n"
            f"images: {names}\n"
        )
        out.images.append(ctx.new_link("images", ".png", text.encode("utf-8")))
        return

    for image_path in image_paths:
        text = (
            f"[emulated $image2image]\n"
            f"prompt: {prompt}\n"
            f"source: {image_path.name}\n"
        )
        out.images.append(ctx.new_link("images", ".png", text.encode("utf-8")))


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.prompts.clear()
    out.images = []

    image_paths = _image_paths(ctx, inp.bundle)
    if not image_paths:
        link = ctx.new_link(
            "images",
            ".png",
            b"[ $image2image: no images[] input ]\n",
        )
        out.images = [link]
        return out

    use_all = _truthy(inp.args.get("use_all", ""))
    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _int_arg(inp.args, "steps", _DEFAULT_STEPS)
    seed = _optional_int(inp.args, "seed")
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_IMAGE2IMAGE_GPU", "1")))
    model_arg = inp.args.get("model", DEFAULT_MODEL)

    if use_all:
        batch = image_paths[:_MAX_USE_ALL_IMAGES]
        if len(image_paths) > _MAX_USE_ALL_IMAGES:
            skipped = len(image_paths) - _MAX_USE_ALL_IMAGES
            print(
                f"$image2image: use_all=True uses first {_MAX_USE_ALL_IMAGES} images "
                f"({skipped} skipped — model supports up to 4)",
                flush=True,
            )
        if len(batch) > _MAX_COMFY_ENCODE_IMAGES:
            print(
                f"$image2image: Comfy encode uses first {_MAX_COMFY_ENCODE_IMAGES} images "
                f"as references (TextEncodeQwenImageEditPlus limit)",
                flush=True,
            )
            batch = batch[:_MAX_COMFY_ENCODE_IMAGES]
        prompts = _prompts_for_images(ctx, inp, 1)
        prompt = prompts[0]
    else:
        batch = None
        prompts = _prompts_for_images(ctx, inp, len(image_paths))

    if _emulate_enabled():
        if use_all:
            _emulate(ctx, out, prompt=prompt, image_paths=batch, use_all=True)
        else:
            for image_path, prompt in zip(image_paths, prompts):
                _emulate(
                    ctx,
                    out,
                    prompt=prompt,
                    image_paths=[image_path],
                    use_all=False,
                )
        return out

    work_dir = ctx.op_dir / "comfy_work"
    jobs: list[tuple[list[Path], str]] = []
    if use_all:
        jobs.append((batch, prompt))
    else:
        jobs.extend(
            ([image_path], job_prompt)
            for image_path, job_prompt in zip(image_paths, prompts)
        )

    run_total = len(jobs)
    t_load = time.perf_counter()
    print(
        "$image2image: building comfy_lib pipeline (first job may load checkpoint)",
        flush=True,
    )

    for run_index, (job_images, job_prompt) in enumerate(jobs):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$image2image cancelled")
        run_seed = seed + run_index if seed is not None else None
        job_width, job_height = resolve_output_size(
            job_images[0], width=width, height=height
        )
        print(
            f"$image2image: edit {run_index + 1}/{run_total} "
            f"({len(job_images)} image(s)) steps={steps} "
            f"size={job_width}x{job_height}",
            flush=True,
        )
        if run_index == 0:
            print(
                f"$image2image: pipeline ready in {time.perf_counter() - t_load:.1f}s",
                flush=True,
            )
        try:
            result = run_comfy_edit(
                work_dir=work_dir,
                image_paths=job_images,
                prompt=job_prompt,
                model_arg=model_arg,
                steps=steps,
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

    return out
