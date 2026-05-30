"""Text generation with local GGUF models via llama-cpp-python."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from externals.llm.model_paths import resolve_gguf_path

_ENGINE_CACHE: dict[tuple, Any] = {}


def _default_gpu_layers() -> int:
    return int(os.environ.get("LLM_N_GPU_LAYERS", "-1"))


def _resolve_rope_scaling_type(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    import llama_cpp

    key = f"LLAMA_ROPE_SCALING_TYPE_{value.upper()}"
    resolved = getattr(llama_cpp, key, None)
    if resolved is None:
        raise ValueError(f"Unknown rope_scaling_type {value!r}")
    return int(resolved)


@dataclass
class GgufLlm:
    """Wrapper around a single GGUF file."""

    name: str
    gguf: str
    n_ctx: int = 4096
    n_gpu_layers: int | None = None
    chat_format: str | None = None
    rope_scaling_type: str | int | None = None
    yarn_orig_ctx: int | None = None
    rope_freq_scale: float | None = None

    def __post_init__(self) -> None:
        if self.n_gpu_layers is None:
            self.n_gpu_layers = _default_gpu_layers()

    def with_overrides(self, **kwargs) -> GgufLlm:
        return replace(self, **kwargs)

    @property
    def gguf_path(self) -> str:
        return resolve_gguf_path(self.gguf)

    def _engine_cache_key(self) -> tuple:
        return (
            self.gguf_path,
            self.n_ctx,
            self.n_gpu_layers,
            self.chat_format,
            self.rope_scaling_type,
            self.yarn_orig_ctx,
            self.rope_freq_scale,
        )

    def _engine(self):
        path = self.gguf_path
        key = self._engine_cache_key()
        if key not in _ENGINE_CACHE:
            from llama_cpp import Llama

            supports_gpu = bool(
                getattr(Llama, "llama_supports_gpu_offload", lambda: False)()
            )
            yarn_note = ""
            if self.yarn_orig_ctx:
                yarn_note = f", yarn_orig_ctx={self.yarn_orig_ctx}"
            print(
                f"$llm load {self.name}: n_ctx={self.n_ctx}, "
                f"n_gpu_layers={self.n_gpu_layers}, "
                f"gpu_offload_supported={supports_gpu}{yarn_note}"
            )

            kwargs: dict[str, Any] = {
                "model_path": path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": False,
            }
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            rope_type = _resolve_rope_scaling_type(self.rope_scaling_type)
            if rope_type is not None:
                kwargs["rope_scaling_type"] = rope_type
            if self.yarn_orig_ctx:
                kwargs["yarn_orig_ctx"] = self.yarn_orig_ctx
            if self.rope_freq_scale is not None:
                kwargs["rope_freq_scale"] = self.rope_freq_scale
            _ENGINE_CACHE[key] = Llama(**kwargs)
        return _ENGINE_CACHE[key]

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: int = 0,
    ) -> str:
        llm = self._engine()
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        try:
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if seed else None,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            # Models without chat template: fall back to plain completion.
            prefix = f"{system.strip()}\n\n" if system.strip() else ""
            full_prompt = f"{prefix}{prompt.strip()}\n"
            out = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if seed else None,
                echo=False,
            )
            return out["choices"][0]["text"].strip()
