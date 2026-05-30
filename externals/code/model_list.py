"""Named GGUF profiles for $code(model='...')."""

from __future__ import annotations

import os

from externals.code.model_paths import MODEL_GGUF, ensure_model
from externals.llm.context_limit import _TEMPLATE_RESERVE_TOKENS, auto_n_ctx, estimate_tokens
from externals.llm.gguf_llm import GgufLlm

_DEFAULT_GGUF = str(MODEL_GGUF)

# Native training window for Qwen2.5-Coder; use YaRN above this (see Qwen HF README).
YARN_ORIG_CTX = 32_768
DEFAULT_MAX_N_CTX = 131_072
DEFAULT_MIN_N_CTX = 4096


def max_code_n_ctx() -> int:
    raw = os.environ.get(
        "AH_CODE_MAX_N_CTX",
        os.environ.get("AH_CODE_N_CTX", str(DEFAULT_MAX_N_CTX)),
    ).strip()
    return int(raw)


def min_code_n_ctx() -> int:
    raw = os.environ.get("AH_CODE_MIN_N_CTX", str(DEFAULT_MIN_N_CTX)).strip()
    return int(raw)


def resolve_code_n_ctx(
    request_text: str,
    max_tokens: int,
    *,
    explicit: int | None = None,
) -> int:
    """Auto-size context from request payload unless n_ctx= was set explicitly."""
    if explicit is not None:
        return explicit
    # Slight over-estimate for chat template / JSON formatting overhead.
    prompt_tokens = int(estimate_tokens(request_text) * 1.05) + 32
    return auto_n_ctx(
        prompt_tokens,
        max_tokens,
        min_ctx=min_code_n_ctx(),
        max_ctx=max_code_n_ctx(),
        reserve=_TEMPLATE_RESERVE_TOKENS,
    )


def _code_profile_kwargs(*, n_ctx: int) -> dict:
    kw: dict = {"n_ctx": n_ctx, "chat_format": "chatml"}
    if n_ctx > YARN_ORIG_CTX:
        kw["rope_scaling_type"] = "yarn"
        kw["yarn_orig_ctx"] = YARN_ORIG_CTX
    return kw


_llms = [
    GgufLlm("default", _DEFAULT_GGUF, n_ctx=YARN_ORIG_CTX, chat_format="chatml"),
    GgufLlm("Qwen2.5-Coder-14B-Instruct", _DEFAULT_GGUF, n_ctx=YARN_ORIG_CTX, chat_format="chatml"),
]

llms: dict[str, GgufLlm] = {m.name: m for m in _llms}


def get_code_llm(
    name: str,
    *,
    n_gpu_layers: int | None = None,
    n_ctx: int,
) -> GgufLlm:
    ensure_model()
    profile_kw = _code_profile_kwargs(n_ctx=n_ctx)
    if name in llms:
        llm = llms[name].with_overrides(**profile_kw)
    elif name.endswith(".gguf") or "/" in name or "\\" in name:
        llm = GgufLlm(name, name, **profile_kw)
    else:
        available = ", ".join(sorted(llms))
        raise KeyError(f"Unknown code model {name!r}. Available: {available}")

    overrides: dict = {}
    if n_gpu_layers is not None:
        overrides["n_gpu_layers"] = n_gpu_layers
    return llm.with_overrides(**overrides) if overrides else llm
