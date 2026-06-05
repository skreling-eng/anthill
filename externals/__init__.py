"""External ($) action handlers — one folder per external, common API."""

from __future__ import annotations

import importlib
from typing import Callable

from externals.api import ExternalContext, ExternalInput
from externals.invoke import subprocess_enabled, write_invoke

_KNOWN = frozenset(
    {
        "file",
        "folder",
        "first_image",
        "input_json",
        "image",
        "image2image",
        "check_image",
        "clear",
        "pass",
        "comfy",
        "image2video",
        "image_clip",
        "video_clip",
        "clip",
        "save",
        "output",
        "sound2text",
        "llm",
        "add_gguf_llm_model",
        "math",
        "code",
        "ah",
        "ah_code_examples",
        "model_ah_create_jsonl",
        "model_ah_train_lora",
        "model_ah_merge_lora",
        "search",
        "serch",
        "texts_to_prompts",
        "texts2prompts",
        "prompts_to_texts",
        "prompts2texts",
        "json2texts",
        "list",
        "music",
        "music_separation",
        "change_voice",
        "draw_text",
        "only",
        "select",
        "join_stems",
        "text2speech",
        "voice_enhance",
        "ocr",
        "image2text",
        "translate",
        "audio_instruct",
        "detach_audio",
        "attach_audio",
        "add_soft_subtitles",
        "video_thumbnailer",
    }
)

# Externals that honor $name(...)[n] via inp.repeat inside the handler (not runtime fan-out).
_REPEAT_NATIVE = frozenset({"image", "image2image", "llm", "math", "music", "code"})

# Externals that read prompts as model input and should not pass them through.
_PROMPT_CONSUMING = frozenset(
    {
        "image",
        "image2image",
        "check_image",
        "image2video",
        "comfy",
        "llm",
        "math",
        "code",
        "search",
        "serch",
        "list",
        "music",
        "prompts_to_texts",
        "prompts2texts",
        "image2text",
        "audio_instruct",
    }
)

_Handler = Callable[[ExternalContext, ExternalInput], "ArrayBundle"]


_EXTERNAL_ALIASES: dict[str, str] = {"serch": "search"}


def _load_handler(name: str) -> _Handler:
    if name in _KNOWN:
        module_name = _EXTERNAL_ALIASES.get(name, name)
        mod = importlib.import_module(f"externals.{module_name}.run")
        return mod.run
    from externals._default.run import run as default_run

    def _wrapped(ctx: ExternalContext, inp: ExternalInput):
        return default_run(ctx, inp, name=name)

    return _wrapped


def run_external(
    name: str, ctx: ExternalContext, inp: ExternalInput
) -> "ArrayBundle":
    """Dispatch to the handler for $name (subprocess by default)."""
    from ahlib.ah_runtime import ArrayBundle, RuntimeCancelled  # noqa: F401

    if ctx.cancel_event is not None and ctx.cancel_event.is_set():
        raise RuntimeCancelled(f"$externals {name!r} cancelled")

    # $ah runs nested Runtime with the parent callback — must stay in-process.
    if subprocess_enabled(name) and name != "ah":
        if name in ("image2image", "image2video"):
            mod = importlib.import_module(f"externals.{name}.worker_client")
            if mod.worker_enabled():
                return mod.run_via_worker(ctx, inp)
        from externals.invoke import run_external_subprocess

        return run_external_subprocess(name, ctx, inp)
    handler = _load_handler(name)
    return handler(ctx, inp)


def external_handles_repeat(name: str) -> bool:
    """True if $name(...)[n] is handled inside the external (not expanded by runtime)."""
    return name in _REPEAT_NATIVE


def external_consumes_prompts(name: str) -> bool:
    """True if this $ external uses up the prompts array (clear on output)."""
    return name in _PROMPT_CONSUMING


__all__ = [
    "ExternalContext",
    "ExternalInput",
    "external_consumes_prompts",
    "external_handles_repeat",
    "run_external",
    "subprocess_enabled",
    "write_invoke",
]
