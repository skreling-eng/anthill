"""$math — math-focused $llm (default: Qwen3.6 UD-Q4_K_M GGUF)."""

from __future__ import annotations

import os
import random

from externals.api import ExternalContext, ExternalInput
from externals.llm.run import (
    _build_prompt,
    _gpu_layers_from_args,
    _variant_count,
)
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_SYSTEM = (
    "You are a precise mathematics assistant. "
    "Reason step by step, use clear notation, and state the final answer explicitly."
)


def _emulate(
    ctx: ExternalContext, inp: ExternalInput, out: ArrayBundle, model: str
) -> ArrayBundle:
    prompt = _build_prompt(inp, ctx)
    system = inp.args.get("system", _DEFAULT_SYSTEM)
    for i in range(_variant_count(inp)):
        content = (
            f"[emulated $math model={model} variant={i}]\n"
            f"system: {system}\n"
            f"prompt:\n{prompt}\n"
        )
        out.texts.append(ctx.new_link("texts", ".txt", content))
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()

    model_name = inp.args.get("model", "default")
    max_tokens = int(inp.args.get("max_tokens", "1024"))
    temperature = float(inp.args.get("temperature", "0.2"))
    seed_base = int(inp.args.get("seed", "0"))
    system = inp.args.get("system", _DEFAULT_SYSTEM) or _DEFAULT_SYSTEM
    prompt = _build_prompt(inp, ctx)

    if not prompt.strip():
        prompt = "(empty prompt)"

    if os.environ.get("AH_EMULATE_MATH", "").lower() in ("1", "true", "yes"):
        return _emulate(ctx, inp, out, model_name)

    try:
        from externals.math.model_list import get_math_llm

        n_ctx_raw = inp.args.get("n_ctx", "").strip()
        n_ctx = int(n_ctx_raw) if n_ctx_raw else None
        llm = get_math_llm(
            model_name,
            n_gpu_layers=_gpu_layers_from_args(inp),
            n_ctx=n_ctx,
        )
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
            out.texts.append(ctx.new_link("texts", ".txt", text + "\n"))
    except ImportError as exc:
        print(
            f"$math fallback to emulate ({exc}). "
            "Install with: uv pip install llama-cpp-python"
        )
        return _emulate(ctx, inp, out, model_name)
    except (KeyError, FileNotFoundError, OSError) as exc:
        print(f"$math error: {exc}")
        raise

    return out
