"""MiDaS depth weights under models/depth/ (lllyasviel/Annotators)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import ensure_anthill_files
from externals.image.model_paths import models_roots

UPSTREAM_REPO = "lllyasviel/Annotators"

DEPTH_PTH = "dpt_hybrid-midas-501f0c75.pt"


def depth_models_dir() -> Path:
    for root in models_roots():
        candidate = root / "depth"
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / "depth"


def model_path() -> Path:
    return depth_models_dir() / DEPTH_PTH


def model_ready() -> bool:
    return model_path().is_file()


def _download_upstream() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "$depth needs huggingface-hub to download models: "
            "uv sync --extra media"
        ) from exc

    dest = depth_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if model_ready():
        return
    print(f"$depth: downloading {UPSTREAM_REPO}/{DEPTH_PTH} -> {dest}", flush=True)
    cached = hf_hub_download(UPSTREAM_REPO, DEPTH_PTH)
    target = model_path()
    if not target.is_file():
        target.write_bytes(Path(cached).read_bytes())


def ensure_model(*, force: bool = False) -> Path:
    """Resolve MiDaS weights; fetch from anthill bundle or upstream HF."""
    if model_ready() and not force:
        return depth_models_dir()

    try:
        ensure_anthill_files([f"depth/{DEPTH_PTH}"], label="$depth", force=force)
    except Exception:
        pass
    if model_ready():
        return depth_models_dir()

    _download_upstream()

    if not model_ready():
        raise FileNotFoundError(
            f"$depth model missing under {depth_models_dir()}: {DEPTH_PTH}. "
            "Run: uv run python tools/download_models.py --upstream-fallback"
        )
    return depth_models_dir()
