"""Text generation with local GGUF models via llama-cpp-python."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from externals.llm.model_paths import resolve_gguf_path

_ENGINE_CACHE: dict[tuple, Any] = {}


def _default_gpu_layers() -> int:
    return int(os.environ.get("LLM_N_GPU_LAYERS", "-1"))


@dataclass
class GgufLlm:
    """Wrapper around a single GGUF file."""

    name: str
    gguf: str
    n_ctx: int = 4096
    n_gpu_layers: int | None = None
    chat_format: str | None = None

    def __post_init__(self) -> None:
        if self.n_gpu_layers is None:
            self.n_gpu_layers = _default_gpu_layers()

    def with_overrides(self, **kwargs) -> GgufLlm:
        return replace(self, **kwargs)

    @property
    def gguf_path(self) -> str:
        return resolve_gguf_path(self.gguf)

    def _engine(self):
        path = self.gguf_path
        key = (path, self.n_ctx, self.n_gpu_layers, self.chat_format)
        if key not in _ENGINE_CACHE:
            from llama_cpp import Llama

            supports_gpu = bool(
                getattr(Llama, "llama_supports_gpu_offload", lambda: False)()
            )
            print(
                f"$llm load {self.name}: n_gpu_layers={self.n_gpu_layers}, "
                f"gpu_offload_supported={supports_gpu}"
            )

            kwargs: dict[str, Any] = {
                "model_path": path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": False,
            }
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
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
