"""Model paths for $voice_enhance (Resemble Enhance)."""

from __future__ import annotations

import os
from pathlib import Path

from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESEMBLE_DIR = _REPO_ROOT / "models" / "resemble-enhance"
HF_REPO = "ResembleAI/resemble-enhance"
ENHANCER_RUN_DIR = RESEMBLE_DIR / "enhancer_stage2"
CHECKPOINT = ENHANCER_RUN_DIR / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"


def enhance_root() -> Path:
    raw = os.environ.get("AH_RESEMBLE_ENHANCE_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    return RESEMBLE_DIR.resolve()


def enhancer_run_dir() -> Path:
    root = enhance_root()
    custom = os.environ.get("AH_RESEMBLE_ENHANCE_RUN_DIR", "").strip()
    if custom:
        return Path(custom).resolve()
    return (root / "enhancer_stage2").resolve()


def checkpoint_ready(run_dir: Path | None = None) -> bool:
    base = run_dir or enhancer_run_dir()
    return (base / "ds" / "G" / "default" / "mp_rank_00_model_states.pt").is_file()


def ensure_models(run_dir: Path | None = None) -> Path:
    """Resolve enhancer weights (models/ or anthill; optional upstream)."""
    base = run_dir or enhancer_run_dir()
    ckpt = base / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if ckpt.is_file():
        return base.resolve()

    ensure_anthill_tree(
        "resemble-enhance",
        ready=lambda: checkpoint_ready(base),
        label="$voice_enhance",
    )
    if checkpoint_ready(base):
        return base.resolve()

    if upstream_fallback_enabled():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "$voice_enhance needs huggingface-hub to download Resemble Enhance weights.\n"
                f"Or clone {HF_REPO} into {enhance_root()}"
            ) from exc

        root = enhance_root()
        root.mkdir(parents=True, exist_ok=True)
        print(f"$voice_enhance: downloading {HF_REPO} -> {root}", flush=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        snapshot_download(
            HF_REPO,
            local_dir=str(root),
            allow_patterns=["enhancer_stage2/**"],
        )

    if not checkpoint_ready(base):
        raise FileNotFoundError(
            f"$voice_enhance: checkpoint missing: {ckpt}\n"
            f"Run: uv run python tools/download_models.py"
        )
    return base.resolve()
