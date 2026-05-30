"""Qwen2-VL-2B-Instruct under models/qwen-vl/."""

from __future__ import annotations

from pathlib import Path

from externals.image.model_paths import models_roots

HF_REPO = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_SUBDIR = Path("qwen-vl") / "Qwen2-VL-2B-Instruct"


def model_dir() -> Path:
    for root in models_roots():
        candidate = root / MODEL_SUBDIR
        if (candidate / "config.json").is_file():
            return candidate
    return models_roots()[0] / MODEL_SUBDIR


def _has_weights(path: Path) -> bool:
    if (path / "model.safetensors").is_file():
        return True
    index = path / "model.safetensors.index.json"
    if index.is_file():
        import json

        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shards = {str(v) for v in weight_map.values()}
        return bool(shards) and all((path / name).is_file() for name in shards)
    return any(path.glob("model-*.safetensors"))


def model_ready() -> bool:
    path = model_dir()
    return (path / "config.json").is_file() and _has_weights(path)


def ensure_model(*, force: bool = False) -> Path:
    """Download Qwen2-VL-2B-Instruct into models/qwen-vl/."""
    path = model_dir()
    path.mkdir(parents=True, exist_ok=True)
    if model_ready() and not force:
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "$image2text needs huggingface-hub to download the model: "
            "uv sync --extra image2text"
        ) from exc

    snapshot_download(HF_REPO, local_dir=str(path))
    if not model_ready():
        raise FileNotFoundError(
            f"Model download finished but {path} is missing config.json or weights"
        )
    return path
