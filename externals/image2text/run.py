"""$image2text — vision-language captioning via Qwen2-VL."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_prompt_texts,
)
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_PROMPT = "Describe this image in detail."


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE2TEXT", "").lower() in (
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
        return [_DEFAULT_PROMPT] * count
    if len(prompts) == 1:
        return [prompts[0]] * count
    if len(prompts) >= count:
        return prompts[:count]
    padded = list(prompts)
    padded.extend([prompts[-1]] * (count - len(prompts)))
    return padded


def _help() -> str:
    return (
        "$image2text requires torch and transformers (media venv).\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "Test without models: AH_EMULATE_IMAGE2TEXT=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()
    out.prompts.clear()

    images = _image_paths(ctx, inp.bundle)
    if not images:
        link = ctx.new_link("texts", ".txt", "[ $image2text: no images[] input ]\n")
        out.texts.append(link)
        return out

    prompts = _prompts_for_images(ctx, inp, len(images))
    max_tokens = _int_arg(inp.args, "max_tokens", 512)
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_IMAGE2TEXT_GPU", "")))

    if _emulate_enabled():
        for image_path, prompt in zip(images, prompts):
            text = (
                f"[emulated $image2text] {image_path.name}\n"
                f"prompt: {prompt}\n"
            )
            out.texts.append(ctx.new_link("texts", ".txt", text))
        return out

    try:
        from externals.image2text.model_paths import ensure_model
        from externals.image2text.qwen_vl import describe_image, load_model
    except ImportError as exc:
        raise RuntimeError(_help()) from exc

    model_dir = ensure_model()
    model, processor = load_model(model_dir, use_gpu=use_gpu)

    for image_path, prompt in zip(images, prompts):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$image2text cancelled")
        text = describe_image(
            model,
            processor,
            image_path,
            prompt,
            max_new_tokens=max_tokens,
        )
        out.texts.append(ctx.new_link("texts", ".txt", text))

    return out
