"""Local code-generation model via llama-cpp-python (brain-local, no anthill imports)."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from brain.config import BrainConfig, YARN_ORIG_CTX, load_config
from brain.llm.context_limit import (
    AgentPromptParts,
    ChatHistory,
    build_chat_messages,
    estimate_messages_chars,
    prompt_char_budget,
    trim_agent_prompt,
    trim_chat_history,
    trim_plain_text,
)

_ENGINE: Any | None = None
_ENGINE_KEY: tuple | None = None


def _clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


@dataclass
class CodeModel:
    config: BrainConfig
    _last_trim_notes: list[str] = field(default_factory=list)

    @property
    def emulate(self) -> bool:
        return self.config.emulate

    @property
    def last_trim_notes(self) -> list[str]:
        return list(self._last_trim_notes)

    def _load_engine(self, n_ctx: int):
        global _ENGINE, _ENGINE_KEY
        path = str(self.config.model_gguf)
        if not path or not self.config.model_gguf.is_file():
            raise FileNotFoundError(
                f"Code model not found: {self.config.model_gguf}. "
                "Download with tools/download_models.py or set BRAIN_EMULATE=1."
            )
        key = (path, n_ctx, self.config.n_gpu_layers)
        if _ENGINE is not None and _ENGINE_KEY == key:
            return _ENGINE
        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": path,
            "n_ctx": n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "chat_format": "chatml",
            "verbose": False,
        }
        if n_ctx > YARN_ORIG_CTX:
            kwargs["rope_scaling_type"] = 1  # yarn
            kwargs["yarn_orig_ctx"] = YARN_ORIG_CTX
        yarn = ", YaRN" if n_ctx > YARN_ORIG_CTX else ""
        try:
            import llama_cpp.llama_cpp as lc

            gpu_ok = bool(getattr(lc, "llama_supports_gpu_offload", lambda: False)())
        except Exception:
            gpu_ok = False
        if not gpu_ok and self.config.n_gpu_layers != 0:
            print(
                "brain: WARNING — llama-cpp-python has no GPU support; "
                "inference will use CPU only (slow). "
                "Reinstall with CUDA: see brain/README.md#gpu",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"brain: loading code model n_ctx={n_ctx} "
            f"n_gpu_layers={self.config.n_gpu_layers}{yarn} "
            f"gpu_offload={gpu_ok}",
            file=sys.stderr,
            flush=True,
        )
        _ENGINE = Llama(**kwargs)
        _ENGINE_KEY = key
        return _ENGINE

    def _prepare_chat(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        history: ChatHistory | None = None,
    ) -> tuple[list[dict[str, str]], int, list[str]]:
        notes: list[str] = []
        pre_budget = prompt_char_budget(self.config.effective_max_ctx(), max_tokens)
        hist_budget = min(
            self.config.conversation_history_chars,
            int(pre_budget * 0.5),
        )
        trimmed_history, hist_notes = trim_chat_history(history or [], budget_chars=hist_budget)
        notes.extend(hist_notes)

        messages = build_chat_messages(prompt, system=system, history=trimmed_history)
        prompt_budget = pre_budget - estimate_messages_chars(messages) + len(prompt)
        if len(prompt) > max(prompt_budget, 512):
            prompt, trim_notes = trim_plain_text(
                prompt, budget_chars=max(prompt_budget, 512), label="prompt"
            )
            notes.extend(trim_notes)
            messages = build_chat_messages(prompt, system=system, history=trimmed_history)

        transcript = "\n".join(m["content"] for m in messages)
        n_ctx = self.config.resolve_n_ctx(transcript, max_tokens)
        budget = prompt_char_budget(n_ctx, max_tokens)
        total = estimate_messages_chars(messages)
        if total > budget:
            shrink = min(
                self.config.conversation_history_chars // 2,
                int(budget * 0.4),
            )
            trimmed_history, more = trim_chat_history(trimmed_history, budget_chars=shrink)
            notes.extend(more)
            messages = build_chat_messages(prompt, system=system, history=trimmed_history)
            total = estimate_messages_chars(messages)
            if total > budget:
                prompt, trim_notes = trim_plain_text(
                    prompt, budget_chars=max(budget // 2, 512), label="prompt (final)"
                )
                notes.extend(trim_notes)
                messages = build_chat_messages(prompt, system=system, history=trimmed_history)

        return messages, n_ctx, notes

    def _prepare_prompt(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
    ) -> tuple[str, int, list[str]]:
        notes: list[str] = []
        pre_budget = prompt_char_budget(self.config.effective_max_ctx(), max_tokens)
        prompt, pre_notes = trim_plain_text(
            prompt, budget_chars=pre_budget, label="prompt (pre-size)"
        )
        notes.extend(pre_notes)

        n_ctx = self.config.resolve_n_ctx(f"{system}\n{prompt}", max_tokens)
        budget = prompt_char_budget(n_ctx, max_tokens)
        if len(prompt) > budget:
            prompt, trim_notes = trim_plain_text(
                prompt, budget_chars=budget, label="prompt"
            )
            notes.extend(trim_notes)
        return prompt, n_ctx, notes

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        history: ChatHistory | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        if self.config.emulate:
            self._last_trim_notes = []
            return self._emulate(prompt, system=system)

        messages, n_ctx, notes = self._prepare_chat(
            prompt, system=system, max_tokens=max_tokens, history=history
        )
        self._last_trim_notes = notes
        for note in notes:
            print(f"brain: {note}", file=sys.stderr, flush=True)

        llm = self._load_engine(n_ctx)
        try:
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            return _clean_text(out["choices"][0]["message"]["content"])
        except Exception:
            prefix = f"{system.strip()}\n\n" if system.strip() else ""
            out = llm(
                f"{prefix}{prompt.strip()}\n",
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                echo=False,
            )
            return _clean_text(out["choices"][0]["text"])

    def complete_agent(
        self,
        parts: AgentPromptParts,
        *,
        system: str = "",
        history: ChatHistory | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Complete with structured agent prompt trimming."""
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        notes: list[str] = []

        pre_budget = prompt_char_budget(self.config.effective_max_ctx(), max_tokens)
        hist_budget = min(
            self.config.conversation_history_chars,
            int(pre_budget * 0.45),
        )
        trimmed_history, hist_notes = trim_chat_history(history or [], budget_chars=hist_budget)
        notes.extend(hist_notes)

        agent_budget = pre_budget - sum(len(u) + len(a) for u, a in trimmed_history)
        trimmed, trim_notes = trim_agent_prompt(
            parts, budget_chars=max(agent_budget, 2048)
        )
        notes.extend(trim_notes)

        prompt = trimmed.render()
        messages = build_chat_messages(prompt, system=system, history=trimmed_history)
        transcript = "\n".join(m["content"] for m in messages)
        n_ctx = self.config.resolve_n_ctx(transcript, max_tokens)
        budget = prompt_char_budget(n_ctx, max_tokens)
        total = estimate_messages_chars(messages)
        if total > budget:
            trimmed, more = trim_agent_prompt(trimmed, budget_chars=budget // 2)
            notes.extend(more)
            prompt = trimmed.render()
            messages = build_chat_messages(prompt, system=system, history=trimmed_history)

        self._last_trim_notes = notes
        for note in notes:
            print(f"brain: {note}", file=sys.stderr, flush=True)

        if self.config.emulate:
            return self._emulate(prompt, system=system)

        llm = self._load_engine(n_ctx)
        temperature = temperature if temperature is not None else self.config.temperature
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        return _clean_text(out["choices"][0]["message"]["content"])

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        history: ChatHistory | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        text = self.complete(
            prompt,
            system=system,
            history=history,
            max_tokens=max_tokens or 2048,
        )
        return _parse_json_object(text)

    @staticmethod
    def _emulate(prompt: str, *, system: str) -> str:
        if "plan" in prompt.lower() or "files_to_read" in prompt:
            return json.dumps(
                {
                    "summary": "[emulated] Analyze request and gather context",
                    "files_to_read": ["_lang_desc", "ahlib/ah_parser.py"],
                    "search_queries": ["anthill .ah language pipeline"],
                    "grep_terms": ["@instruction", "ExternalContext"],
                },
                indent=2,
            )
        return (
            "[emulated brain output]\n"
            f"system: {system[:80]}...\n\n"
            "--- a/ahlib/ah_parser.py\n"
            "+++ b/ahlib/ah_parser.py\n"
            "@@ -1,3 +1,4 @@\n"
            " # example unified diff (not applied)\n"
            "+# change requested by user\n"
        )


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object from model")
    return data


def get_code_model(config: BrainConfig | None = None) -> CodeModel:
    return CodeModel(config or load_config())
