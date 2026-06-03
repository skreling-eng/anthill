"""Qwen3.6-35B-A3B UD-Q4_K_M GGUF for $math (unsloth bundle)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import (
    require_models_file,
    resolve_models_file,
    upstream_fallback_enabled,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _REPO_ROOT / "models"
MODEL_DIR = MODELS_DIR / "math" / "Qwen3.6-35B-A3B-GGUF"
MODEL_GGUF = MODEL_DIR / "model.gguf"
ANTHILL_GGUF = "math/Qwen3.6-35B-A3B-GGUF/model.gguf"

HF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
HF_GGUF = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def model_ready() -> bool:
    return MODEL_GGUF.is_file()


def _download_from_unsloth() -> Path:
    import shutil

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "$math needs huggingface-hub to download the model: "
            "uv pip install huggingface-hub"
        ) from exc

    print(
        f"$math: downloading {HF_GGUF} from {HF_REPO} (~23 GB)",
        flush=True,
    )
    downloaded = Path(hf_hub_download(HF_REPO, HF_GGUF, local_dir=str(MODEL_DIR)))
    if downloaded.resolve() != MODEL_GGUF.resolve():
        shutil.copy2(downloaded, MODEL_GGUF)
    return MODEL_GGUF


def ensure_model(*, force: bool = False) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_ready() and not force:
        return MODEL_GGUF

    found = resolve_models_file(ANTHILL_GGUF)
    if found is not None and found.is_file():
        if found.resolve() != MODEL_GGUF.resolve():
            import shutil

            shutil.copy2(found, MODEL_GGUF)
        return MODEL_GGUF

    if upstream_fallback_enabled():
        return _download_from_unsloth()

    path = require_models_file(ANTHILL_GGUF, label="$math")
    if path.resolve() != MODEL_GGUF.resolve():
        import shutil

        shutil.copy2(path, MODEL_GGUF)
    if not model_ready():
        raise FileNotFoundError(
            f"Model not found: {MODEL_GGUF}. "
            "Run: uv run python tools/download_models.py --upstream-fallback"
        )
    return MODEL_GGUF
