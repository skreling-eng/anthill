"""Named GGUF profiles for $code(model='...') — built from CODE_MODELS registry."""

from __future__ import annotations

import os

from externals.code.model_paths import (
    CODE_MODELS,
    ensure_model,
    get_code_profile,
    resolve_profile_key,
)
from externals.llm.context_limit import _TEMPLATE_RESERVE_TOKENS, auto_n_ctx, estimate_tokens
from externals.llm.gguf_llm import GgufLlm

# Native training window for Qwen2.5-Coder; use YaRN above this (see Qwen HF README).
YARN_ORIG_CTX = 32_768
DEFAULT_MAX_N_CTX = 131_072
DEFAULT_MIN_N_CTX = 4096
# Default auto-size cap (trim + KV); 16k is much faster than 32k on 16GB VRAM.
DEFAULT_AUTO_MAX_N_CTX = 16_384


def max_code_n_ctx() -> int:
    raw = os.environ.get(
        "AH_CODE_MAX_N_CTX",
        os.environ.get("AH_CODE_N_CTX", str(DEFAULT_MAX_N_CTX)),
    ).strip()
    return int(raw)


def min_code_n_ctx() -> int:
    raw = os.environ.get("AH_CODE_MIN_N_CTX", str(DEFAULT_MIN_N_CTX)).strip()
    return int(raw)


def auto_max_code_n_ctx() -> int:
    """Upper bound for auto n_ctx (smaller = faster; request is trimmed to fit)."""
    raw = os.environ.get("AH_CODE_AUTO_MAX_N_CTX", str(DEFAULT_AUTO_MAX_N_CTX)).strip()
    try:
        n = int(raw)
    except ValueError:
        n = DEFAULT_AUTO_MAX_N_CTX
    return max(min_code_n_ctx(), n)


def extended_code_ctx_enabled() -> bool:
    """Allow auto n_ctx above YARN_ORIG_CTX (YaRN; llama.cpp train-overflow warning)."""
    raw = os.environ.get("AH_CODE_EXTENDED_CTX", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def resolve_code_n_ctx(
    request_text: str,
    max_tokens: int,
    *,
    explicit: int | None = None,
) -> int:
    """Auto-size context from request payload unless n_ctx= was set explicitly."""
    if explicit is not None:
        return explicit
    prompt_tokens = int(estimate_tokens(request_text) * 1.05) + 32
    max_ctx = max_code_n_ctx()
    if not extended_code_ctx_enabled():
        max_ctx = min(max_ctx, YARN_ORIG_CTX, auto_max_code_n_ctx())
    ctx = auto_n_ctx(
        prompt_tokens,
        max_tokens,
        min_ctx=min_code_n_ctx(),
        max_ctx=max_ctx,
        reserve=_TEMPLATE_RESERVE_TOKENS,
    )
    return ctx


def _code_profile_kwargs(*, n_ctx: int) -> dict:
    kw: dict = {"n_ctx": n_ctx, "chat_format": "chatml"}
    if n_ctx > YARN_ORIG_CTX:
        kw["rope_scaling_type"] = "yarn"
        kw["yarn_orig_ctx"] = YARN_ORIG_CTX
    return kw


def _build_llms() -> dict[str, GgufLlm]:
    out: dict[str, GgufLlm] = {}
    for profile in CODE_MODELS.values():
        gguf = str(profile.model_gguf)
        for alias in profile.aliases:
            out[alias] = GgufLlm(
                alias,
                gguf,
                n_ctx=YARN_ORIG_CTX,
                chat_format=profile.chat_format,
            )
    return out


llms: dict[str, GgufLlm] = _build_llms()


def get_code_llm(
    name: str,
    *,
    n_gpu_layers: int | None = None,
    n_ctx: int,
) -> GgufLlm:
    profile_kw = _code_profile_kwargs(n_ctx=n_ctx)

    if name.endswith(".gguf") or "/" in name or "\\" in name:
        llm = GgufLlm(name, name, **profile_kw)
    elif name in llms:
        profile_key = resolve_profile_key(name)
        ensure_model(key=profile_key)
        llm = llms[name].with_overrides(**profile_kw)
    else:
        try:
            profile_key = resolve_profile_key(name)
        except KeyError:
            available = ", ".join(sorted(llms))
            raise KeyError(f"Unknown code model {name!r}. Available: {available}") from None
        ensure_model(key=profile_key)
        profile = get_code_profile(name)
        llm = GgufLlm(name, str(profile.model_gguf), **profile_kw)

    overrides: dict = {}
    if n_gpu_layers is not None:
        overrides["n_gpu_layers"] = n_gpu_layers
    return llm.with_overrides(**overrides) if overrides else llm
