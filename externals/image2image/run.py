"""$image2image — Qwen-Rapid-AIO prompt-guided image edit (comfy_lib in-process)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.image2image.comfy_workflow import resolve_output_size
from externals.image2image.comfy_runner import (
    run_comfy_edit,
    run_comfy_edit_variants,
    save_png_bytes,
)
from externals.image2image.model_paths import DEFAULT_MODEL, available_models, is_klein_model
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_STEPS = 4
_DEFAULT_KLEIN_STEPS = 4
_MAX_USE_ALL_IMAGES = 4
_MAX_COMFY_ENCODE_IMAGES = 3


def _image_file_prefix(model_arg: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w.-]", "_", model_arg)
    return f"{ts}_{safe_model}_"


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


def _repeat_count(inp: ExternalInput) -> int:
    """Variants per (image, prompt) pair ($image2image(...)[n])."""
    return max(1, inp.repeat) if inp.repeat > 0 else 1


def _read_prompt_list(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    """Read prompts[] / prompt_text once per $image2image invocation."""
    arg_prompt = inp.args.get("prompt", "").strip()
    if arg_prompt:
        return [arg_prompt]
    return read_prompt_texts(ctx, inp)


def _job_prompts(prompts: list[str]) -> list[str]:
    """At least one prompt slot so every image still runs when prompts[] is empty."""
    return prompts if prompts else [""]


def _build_edit_jobs(
    *,
    image_paths: list[Path],
    prompts: list[str],
    use_all: bool,
    repeat: int,
) -> list[tuple[list[Path], str]]:
    """(image batch, prompt) jobs: images × prompts × repeat (use_all: one batch per prompt)."""
    jobs: list[tuple[list[Path], str]] = []
    prompt_list = _job_prompts(prompts)
    if use_all:
        batch = image_paths[:_MAX_USE_ALL_IMAGES]
        if len(batch) > _MAX_COMFY_ENCODE_IMAGES:
            batch = batch[:_MAX_COMFY_ENCODE_IMAGES]
        for job_prompt in prompt_list:
            for _ in range(repeat):
                jobs.append((batch, job_prompt))
        return jobs
    for image_path in image_paths:
        for job_prompt in prompt_list:
            for _ in range(repeat):
                jobs.append(([image_path], job_prompt))
    return jobs


def _group_edit_jobs(
    jobs: list[tuple[list[Path], str]],
) -> list[tuple[list[Path], str, int]]:
    """Merge identical (image batch, prompt) jobs; preserve first-seen order."""
    groups: list[tuple[list[Path], str, int]] = []
    index: dict[tuple[tuple[str, ...], str], int] = {}
    for job_images, job_prompt in jobs:
        key = (tuple(str(p.resolve()) for p in job_images), job_prompt)
        if key in index:
            gi = index[key]
            imgs, prompt, count = groups[gi]
            groups[gi] = (imgs, prompt, count + 1)
        else:
            index[key] = len(groups)
            groups.append((job_images, job_prompt, 1))
    return groups


def _help() -> str:
    return (
        "$image2image uses comfy_lib (in-process).\n"
        "  Qwen-Rapid: models/qwen-rapid/ (sfw-v23, nsfw-v23)\n"
        "  Flux.2 Klein FP8: models/flux2klein/flux2Klein9bFp8_fp8.safetensors "
        f"(model=klein-fp8; also needs text_encoders/qwen_3_8b_fp8mixed + vae/flux2-vae)\n"
        f"  Checkpoints: {available_models()}\n"
        "  Worker venv: AH_EXTERNAL_VENV_image2image=.venvs/media (comfy-kitchen for FP8).\n"
        "Test without GPU/model: AH_EMULATE_IMAGE2IMAGE=1"
    )


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    prompt: str,
    image_paths: list[Path],
    use_all: bool,
    file_prefix: str,
) -> None:
    if use_all:
        names = ", ".join(p.name for p in image_paths[:_MAX_USE_ALL_IMAGES])
        text = (
            f"[emulated $image2image use_all=True]\n"
            f"prompt: {prompt}\n"
            f"images: {names}\n"
        )
        out.images.append(
            ctx.new_link("images", ".png", text.encode("utf-8"), prefix=file_prefix)
        )
        return

    for image_path in image_paths:
        text = (
            f"[emulated $image2image]\n"
            f"prompt: {prompt}\n"
            f"source: {image_path.name}\n"
        )
        out.images.append(
            ctx.new_link("images", ".png", text.encode("utf-8"), prefix=file_prefix)
        )


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
    model_arg = inp.args.get("model", DEFAULT_MODEL)
    steps = _int_arg(inp.args, "steps", _DEFAULT_KLEIN_STEPS if is_klein_model(model_arg) else _DEFAULT_STEPS)
    seed = _optional_int(inp.args, "seed")
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_IMAGE2IMAGE_GPU", "1")))
    repeat = _repeat_count(inp)
    prompt_list = _read_prompt_list(ctx, inp)
    file_prefix = _image_file_prefix(model_arg)

    if use_all and is_klein_model(model_arg):
        print(
            "$image2image: Flux.2 Klein edit uses the first input image only (use_all ignored)",
            flush=True,
        )
    if use_all and len(image_paths) > _MAX_USE_ALL_IMAGES:
        skipped = len(image_paths) - _MAX_USE_ALL_IMAGES
        print(
            f"$image2image: use_all=True uses first {_MAX_USE_ALL_IMAGES} images "
            f"({skipped} skipped — model supports up to 4)",
            flush=True,
        )
    if use_all and len(image_paths) > _MAX_COMFY_ENCODE_IMAGES:
        print(
            f"$image2image: Comfy encode uses first {_MAX_COMFY_ENCODE_IMAGES} images "
            f"as references (TextEncodeQwenImageEditPlus limit)",
            flush=True,
        )

    jobs = _build_edit_jobs(
        image_paths=image_paths,
        prompts=prompt_list,
        use_all=use_all,
        repeat=repeat,
    )

    if _emulate_enabled():
        for job_images, job_prompt in jobs:
            _emulate(
                ctx,
                out,
                prompt=job_prompt,
                image_paths=job_images,
                use_all=use_all and len(job_images) > 1,
                file_prefix=file_prefix,
            )
        return out

    work_dir = ctx.op_dir / "comfy_work"
    if repeat > 1:
        print(
            f"$image2image: repeat={repeat} → {len(jobs)} edit job(s) "
            f"(images × prompts × repeat)",
            flush=True,
        )

    groups = _group_edit_jobs(jobs)
    run_total = len(jobs)
    t_load = time.perf_counter()
    print(
        "$image2image: building comfy_lib pipeline (first job may load checkpoint)",
        flush=True,
    )

    global_idx = 0
    for group_idx, (job_images, job_prompt, variant_count) in enumerate(groups):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$image2image cancelled")
        job_width, job_height = resolve_output_size(
            job_images[0], width=width, height=height
        )
        fast = variant_count > 1
        print(
            f"$image2image: edit {group_idx + 1}/{len(groups)} "
            f"({len(job_images)} image(s)"
            f"{f', {variant_count} variants' if fast else ''}) "
            f"steps={steps} size={job_width}x{job_height} "
            f"prompt={job_prompt[:80]!r}",
            flush=True,
        )
        if group_idx == 0:
            print(
                f"$image2image: pipeline ready in {time.perf_counter() - t_load:.1f}s",
                flush=True,
            )
        try:
            if fast:
                seeds = [
                    seed + global_idx + i if seed is not None else None
                    for i in range(variant_count)
                ]

                def _save_variant(pil) -> None:
                    out.images.append(
                        ctx.new_link(
                            "images",
                            ".png",
                            save_png_bytes(pil),
                            prefix=file_prefix,
                        )
                    )
                    (ctx.op_dir / "output.json").write_text(
                        json.dumps(out.as_dict(), indent=2),
                        encoding="utf-8",
                    )

                run_comfy_edit_variants(
                    work_dir=work_dir,
                    image_paths=job_images,
                    prompt=job_prompt,
                    model_arg=model_arg,
                    steps=steps,
                    seeds=seeds,
                    width=job_width,
                    height=job_height,
                    use_gpu=use_gpu,
                    on_variant=_save_variant,
                )
            else:
                run_seed = seed + global_idx if seed is not None else None
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
                out.images.append(
                    ctx.new_link(
                        "images",
                        ".png",
                        save_png_bytes(result),
                        prefix=file_prefix,
                    )
                )
        except ImportError as exc:
            raise RuntimeError(_help()) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        global_idx += variant_count

    return out
