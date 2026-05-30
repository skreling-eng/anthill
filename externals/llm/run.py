"""$llm — text-to-text via local GGUF models (llama-cpp-python)."""

from __future__ import annotations

import os
import random

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_CONTENT_MARKER = "__CONTENT__"


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _add_texts_enabled(inp: ExternalInput) -> bool:
    return _truthy(inp.args.get("add_texts", ""))


def _collect_texts(ctx: ExternalContext, inp: ExternalInput) -> str:
    parts: list[str] = []
    for link in inp.bundle.texts:
        text = ctx.read_link_text(link)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _build_prompt(inp: ExternalInput, ctx: ExternalContext) -> str:
    parts: list[str] = []
    if inp.prompt_text.strip():
        parts.append(inp.prompt_text.strip())
    if not _add_texts_enabled(inp):
        for link in inp.bundle.texts:
            text = ctx.read_link_text(link)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    prompt = "\n\n".join(parts)
    content = _collect_texts(ctx, inp)
    if content:
        if prompt:
            return f"{prompt}\n\n{_CONTENT_MARKER}\n{content}"
        return f"{_CONTENT_MARKER}\n{content}"
    return prompt


def _variant_count(inp: ExternalInput) -> int:
    if inp.repeat > 1:
        return inp.repeat
    return max(1, int(inp.args.get("count", "1")))


def _gpu_layers_from_args(inp: ExternalInput) -> int | None:
    raw = inp.args.get("gpu_layers", inp.args.get("n_gpu_layers"))
    if raw is None or raw == "":
        return None
    return int(raw)


def _emulate(
    ctx: ExternalContext, inp: ExternalInput, out: ArrayBundle, model: str
) -> ArrayBundle:
    prompt = _build_prompt(inp, ctx)
    system = inp.args.get("system", "")
    for i in range(_variant_count(inp)):
        content = (
            f"[emulated $llm model={model} variant={i}]\n"
            f"system: {system}\n"
            f"prompt:\n{prompt}\n"
        )
        link = ctx.new_link("texts", ".txt", content)
        out.texts.append(link)
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()
    model_name = inp.args.get("model", "default")
    max_tokens = int(inp.args.get("max_tokens", "512"))
    temperature = float(inp.args.get("temperature", "0.7"))
    seed_base = int(inp.args.get("seed", "0"))
    system = inp.args.get("system", "")
    prompt = _build_prompt(inp, ctx)

    if not prompt.strip():
        prompt = "(empty prompt)"

    if os.environ.get("AH_EMULATE_LLM", "").lower() in ("1", "true", "yes"):
        return _emulate(ctx, inp, out, model_name)

    try:
        from externals.llm.model_list import get_llm

        llm = get_llm(model_name, n_gpu_layers=_gpu_layers_from_args(inp))
        count = _variant_count(inp)
        for i in range(count):
            seed = seed_base if seed_base else random.randint(0, 2**31 - 1)
            if count > 1 and seed_base:
                seed = seed_base + i
            text = llm.complete(
                prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            link = ctx.new_link("texts", ".txt", text + "\n")
            out.texts.append(link)
    except ImportError as exc:
        print(
            f"$llm fallback to emulate ({exc}). "
            "Install with: uv pip install llama-cpp-python"
        )
        return _emulate(ctx, inp, out, model_name)
    except (KeyError, FileNotFoundError, OSError) as exc:
        print(f"$llm error: {exc}")
        raise

    return out
