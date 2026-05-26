"""Named GGUF LLM profiles for $llm(model='...')."""

from __future__ import annotations

from externals.llm.gguf_llm import GgufLlm
from externals.llm.model_paths import _GEMMA_DEFAULT_DIR

_GEMMA_GGUF = f"{_GEMMA_DEFAULT_DIR}/model.gguf"

_llms = [
    GgufLlm(
        "default",
        _GEMMA_GGUF,
        n_ctx=8192,
        chat_format="gemma",
    ),
    GgufLlm(
        "gemma4",
        _GEMMA_GGUF,
        n_ctx=8192,
        chat_format="gemma",
    ),
    GgufLlm(
        "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS",
        _GEMMA_GGUF,
        n_ctx=8192,
        chat_format="gemma",
    ),
]

llms: dict[str, GgufLlm] = {m.name: m for m in _llms}


def get_llm(
    name: str,
    *,
    n_gpu_layers: int | None = None,
    n_ctx: int | None = None,
) -> GgufLlm:
    if name in llms:
        llm = llms[name]
    elif name.endswith(".gguf") or "/" in name or "\\" in name:
        llm = GgufLlm(name, name, chat_format="gemma")
    else:
        available = ", ".join(sorted(llms))
        raise KeyError(f"Unknown LLM model {name!r}. Available: {available}")

    overrides: dict = {}
    if n_gpu_layers is not None:
        overrides["n_gpu_layers"] = n_gpu_layers
    if n_ctx is not None:
        overrides["n_ctx"] = n_ctx
    return llm.with_overrides(**overrides) if overrides else llm
