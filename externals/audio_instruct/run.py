"""$audio_instruct — audio + prompt → text via Qwen2-Audio 7B Instruct (4-bit)."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_arg_list,
    read_prompt_texts,
)
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_PROMPT = "What do you hear in this audio?"


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_AUDIO_INSTRUCT", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _resolve_use_gpu(inp: ExternalInput) -> bool:
    raw = inp.args.get("gpu", "").strip()
    if raw:
        return _truthy(raw)
    env = os.environ.get("AH_AUDIO_INSTRUCT_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _sound_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.sounds:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _prompts_for_audio(ctx: ExternalContext, inp: ExternalInput, count: int) -> list[str]:
    if inp.args.get("prompt", "").strip():
        return [inp.args["prompt"].strip()] * count
    prompts = read_prompt_texts(ctx, inp)
    if not prompts:
        if inp.prompt_text.strip():
            return [inp.prompt_text.strip()] * count
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
        "$audio_instruct requires torch, transformers, bitsandbytes, and librosa.\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media)\n"
        "  uv run python tools/download_models.py --upstream-fallback  (first run auto-downloads from HF)\n"
        "  CUDA GPU required (4-bit model).\n"
        "Test without models: AH_EMULATE_AUDIO_INSTRUCT=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()
    out.prompts.clear()
    # Keep sounds[] so downstream steps can still see the source clip if needed.

    sounds = _sound_paths(ctx, inp.bundle)
    if not sounds:
        link = ctx.new_link(
            "texts", ".txt", "[ $audio_instruct: no sounds[] input ]\n"
        )
        out.texts.append(link)
        return out

    prompts = _prompts_for_audio(ctx, inp, len(sounds))
    max_tokens = _int_arg(inp.args, "max_tokens", 1024)
    use_gpu = _resolve_use_gpu(inp)
    system = inp.args.get("system", "You are a helpful assistant.").strip()
    model_name = read_arg_list(inp, "model", "default")[0]

    if _emulate_enabled():
        for sound_path, prompt in zip(sounds, prompts):
            text = (
                f"[emulated $audio_instruct] {sound_path.name}\n"
                f"prompt: {prompt}\n"
            )
            out.texts.append(ctx.new_link("texts", ".txt", text))
        return out

    try:
        from externals.audio_instruct.model_list import get_audio_instruct_model
        from externals.audio_instruct.model_paths import ensure_model
        from externals.audio_instruct.qwen_audio import answer_audio, load_model
    except ImportError as exc:
        raise RuntimeError(_help()) from exc

    try:
        profile = get_audio_instruct_model(model_name)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc

    if not use_gpu:
        raise RuntimeError(
            "$audio_instruct requires a CUDA GPU. Pass gpu=True or set AH_AUDIO_INSTRUCT_GPU=1."
        )

    weights = ensure_model(profile)
    model, processor = load_model(weights, use_gpu=use_gpu)

    for sound_path, prompt in zip(sounds, prompts):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$audio_instruct cancelled")
        text = answer_audio(
            model,
            processor,
            sound_path,
            prompt,
            system=system,
            max_new_tokens=max_tokens,
        )
        out.texts.append(ctx.new_link("texts", ".txt", text))

    return out
