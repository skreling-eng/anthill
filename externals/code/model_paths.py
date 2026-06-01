"""Qwen2.5-Coder-14B-Instruct GGUF under models/code/."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import require_models_file, upstream_fallback_enabled

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _REPO_ROOT / "models"
MODEL_DIR = MODELS_DIR / "code" / "Qwen2.5-Coder-14B-Instruct"
MODEL_GGUF = MODEL_DIR / "model.gguf"
ANTHILL_GGUF = "code/Qwen2.5-Coder-14B-Instruct/model.gguf"

HF_REPO = "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF"
HF_GGUF = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"


def model_ready() -> bool:
    return MODEL_GGUF.is_file()


def ensure_model(*, force: bool = False) -> Path:
    """Resolve Qwen2.5-Coder GGUF (models/ or anthill bundle; optional upstream)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_ready() and not force:
        return MODEL_GGUF

    path = require_models_file(ANTHILL_GGUF, label="$code")
    if path.resolve() == MODEL_GGUF.resolve() or model_ready():
        return MODEL_GGUF

    if upstream_fallback_enabled():
        import shutil

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "$code needs huggingface-hub to download the model: "
                "uv pip install huggingface-hub"
            ) from exc

        downloaded = Path(
            hf_hub_download(HF_REPO, HF_GGUF, local_dir=str(MODEL_DIR))
        )
        if downloaded.resolve() != MODEL_GGUF.resolve():
            shutil.copy2(downloaded, MODEL_GGUF)

    if not model_ready():
        raise FileNotFoundError(
            f"Model not found: {MODEL_GGUF}. "
            f"Run: uv run python tools/download_models.py"
        )
    return MODEL_GGUF
