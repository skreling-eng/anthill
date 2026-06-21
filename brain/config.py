"""Brain configuration — paths and model settings (self-contained)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BRAIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BRAIN_DIR.parent

YARN_ORIG_CTX = 32_768
DEFAULT_MAX_N_CTX = 131_072
DEFAULT_MIN_N_CTX = 4096
DEFAULT_AUTO_MAX_N_CTX = 16_384

DEFAULT_CODE_EXTENSIONS = (
    ".ah",
    ".py",
    ".md",
    ".txt",
    ".json",
    ".html",
    ".css",
    ".js",
    ".toml",
    ".yaml",
    ".yml",
)
DEFAULT_SPECIAL_NAMES = (
    "_description",
    "_lang_desc",
    "AH_CODEGEN_INSTRUCTIONS.md",
)
DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    ".venvs",
    ".cache",
    "__pycache__",
    "node_modules",
    "sessions",
    "saves",
    "models",
    "comfy_lib",
    "test_data",
    "brain/sessions",
}


@dataclass
class BrainConfig:
    """Runtime configuration for the brain agent."""

    codebase_root: Path = field(default_factory=lambda: _REPO_ROOT)
    brain_dir: Path = field(default_factory=lambda: _BRAIN_DIR)
    sessions_dir: Path = field(default_factory=lambda: _BRAIN_DIR / "sessions")
    model_name: str = "default"
    model_gguf: Path | None = None
    n_ctx: int | None = None
    auto_max_n_ctx: int = DEFAULT_AUTO_MAX_N_CTX
    max_n_ctx: int = DEFAULT_MAX_N_CTX
    min_n_ctx: int = DEFAULT_MIN_N_CTX
    extended_ctx: bool = False
    max_tokens: int = 4096
    plan_max_tokens: int = 1024
    temperature: float = 0.2
    n_gpu_layers: int | None = None
    emulate: bool = False
    code_extensions: tuple[str, ...] = DEFAULT_CODE_EXTENSIONS
    special_names: tuple[str, ...] = DEFAULT_SPECIAL_NAMES
    skip_dirs: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_SKIP_DIRS))
    max_file_bytes: int = 256 * 1024
    max_files_in_tree: int = 4000
    max_context_files: int = 8
    max_conversation_turns: int = 8
    conversation_history_chars: int = 12_000
    search_limit: int = 5

    def __post_init__(self) -> None:
        self.codebase_root = Path(self.codebase_root).resolve()
        self.brain_dir = Path(self.brain_dir).resolve()
        self.sessions_dir = Path(self.sessions_dir).resolve()
        if self.model_gguf is None:
            self.model_gguf = self._default_model_path()
        if self.n_gpu_layers is None:
            raw = os.environ.get("BRAIN_N_GPU_LAYERS", os.environ.get("LLM_N_GPU_LAYERS", "-1"))
            self.n_gpu_layers = int(raw)
        if os.environ.get("BRAIN_EMULATE", "").lower() in ("1", "true", "yes"):
            self.emulate = True

    def _default_model_path(self) -> Path:
        override = os.environ.get("BRAIN_MODEL_GGUF", "").strip()
        if override:
            return Path(override).resolve()
        return (
            _REPO_ROOT
            / "models"
            / "code"
            / "Qwen2.5-Coder-14B-Instruct"
            / "model.gguf"
        )

    def effective_max_ctx(self) -> int:
        cap = self.max_n_ctx
        if not self.extended_ctx:
            cap = min(cap, YARN_ORIG_CTX, self.auto_max_n_ctx)
        return cap

    def resolve_n_ctx(self, prompt: str, max_tokens: int) -> int:
        """Auto-size context unless n_ctx was set explicitly."""
        if self.n_ctx is not None:
            return self.n_ctx
        from brain.llm.context_limit import auto_n_ctx, estimate_tokens

        prompt_tokens = int(estimate_tokens(prompt) * 1.05) + 32
        return auto_n_ctx(
            prompt_tokens,
            max_tokens,
            min_ctx=self.min_n_ctx,
            max_ctx=self.effective_max_ctx(),
        )

    def ensure_sessions_dir(self) -> Path:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        return self.sessions_dir


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_config() -> BrainConfig:
    kwargs: dict = {}
    root = os.environ.get("BRAIN_CODEBASE_ROOT", "").strip()
    if root:
        kwargs["codebase_root"] = Path(root)
    model = os.environ.get("BRAIN_MODEL", "").strip()
    if model:
        kwargs["model_name"] = model

    explicit_n_ctx = _env_int("BRAIN_N_CTX")
    if explicit_n_ctx is not None:
        kwargs["n_ctx"] = explicit_n_ctx

    for key, env in (
        ("auto_max_n_ctx", "BRAIN_AUTO_MAX_N_CTX"),
        ("max_n_ctx", "BRAIN_MAX_N_CTX"),
        ("min_n_ctx", "BRAIN_MIN_N_CTX"),
        ("max_tokens", "BRAIN_MAX_TOKENS"),
        ("plan_max_tokens", "BRAIN_PLAN_MAX_TOKENS"),
        ("max_context_files", "BRAIN_MAX_CONTEXT_FILES"),
        ("max_conversation_turns", "BRAIN_MAX_CONVERSATION_TURNS"),
        ("conversation_history_chars", "BRAIN_CONVERSATION_HISTORY_CHARS"),
    ):
        val = _env_int(env)
        if val is not None:
            kwargs[key] = val

    if _env_bool("BRAIN_EXTENDED_CTX"):
        kwargs["extended_ctx"] = True

    return BrainConfig(**kwargs)
