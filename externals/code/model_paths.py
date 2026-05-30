"""Qwen2.5-Coder-14B-Instruct GGUF under models/code/."""

from __future__ import annotations

import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _REPO_ROOT / "models"
MODEL_DIR = MODELS_DIR / "code" / "Qwen2.5-Coder-14B-Instruct"
MODEL_GGUF = MODEL_DIR / "model.gguf"

HF_REPO = "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF"
HF_GGUF = "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf"


def model_ready() -> bool:
    return MODEL_GGUF.is_file()


def ensure_model(*, force: bool = False) -> Path:
    """Download Qwen2.5-Coder-14B-Instruct Q4_K_M into models/code/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_ready() and not force:
        return MODEL_GGUF

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
            f"Model download finished but {MODEL_GGUF} is missing"
        )
    return MODEL_GGUF
