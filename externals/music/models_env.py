"""Route ACE-Step / Hugging Face downloads under project models/."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _PROJECT_ROOT / "models"

# Native PyTorch checkpoints (HF layout: acestep-v15-turbo/, vae/, Qwen3-Embedding-0.6B/, …)
DEFAULT_CHECKPOINTS_DIR = MODELS_DIR / "ace-step-1.5_st"
# GGUF weights for ace-synth
DEFAULT_GGUF_DIR = MODELS_DIR / "ace-step-1.5"


def configure_models_environment() -> Path:
    """Set default env vars so all model downloads land in models/ (only if unset)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hf_home = MODELS_DIR / "huggingface"
    hub = hf_home / "hub"

    defaults = {
        "MODELS_PATH": str(MODELS_DIR),
        "ACESTEP_CHECKPOINTS_DIR": str(DEFAULT_CHECKPOINTS_DIR),
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    DEFAULT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
