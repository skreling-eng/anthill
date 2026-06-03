"""Named GGUF profiles for $math(model='...')."""

from __future__ import annotations

from externals.llm.gguf_llm import GgufLlm
from externals.math.model_paths import MODEL_GGUF, ensure_model

_DEFAULT_GGUF = str(MODEL_GGUF)

_llms = [
    GgufLlm(
        "default",
        _DEFAULT_GGUF,
        n_ctx=8192,
        chat_format="chatml",
    ),
    GgufLlm(
        "qwen36",
        _DEFAULT_GGUF,
        n_ctx=8192,
        chat_format="chatml",
    ),
    GgufLlm(
        "Qwen3.6-35B-A3B-UD-Q4_K_M",
        _DEFAULT_GGUF,
        n_ctx=8192,
        chat_format="chatml",
    ),
]

llms: dict[str, GgufLlm] = {m.name: m for m in _llms}


def get_math_llm(
    name: str,
    *,
    n_gpu_layers: int | None = None,
    n_ctx: int | None = None,
) -> GgufLlm:
    ensure_model()
    if name in llms:
        llm = llms[name]
    elif name.endswith(".gguf") or "/" in name or "\\" in name:
        llm = GgufLlm(name, name, n_ctx=8192, chat_format="chatml")
    else:
        available = ", ".join(sorted(llms))
        raise KeyError(f"Unknown math model {name!r}. Available: {available}")

    overrides: dict = {}
    if n_gpu_layers is not None:
        overrides["n_gpu_layers"] = n_gpu_layers
    if n_ctx is not None:
        overrides["n_ctx"] = n_ctx
    return llm.with_overrides(**overrides) if overrides else llm
