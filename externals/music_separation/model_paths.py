"""Intel HTDemucs v4 OpenVINO model (same as openvino-plugins-ai-audacity)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from externals.anthill_models import ensure_anthill_files, upstream_fallback_enabled

_REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _REPO_ROOT / "models"

# HTDemucs v4 — the "large" 4-stem model used by Intel's Audacity Music Separation effect.
# https://github.com/intel/openvino-plugins-ai-audacity (loads openvino-models/htdemucs_v4.xml)
MODEL_ID = "htdemucs_v4"
MODEL_DIR = MODELS_DIR / "demucs-openvino" / MODEL_ID
MODEL_XML = MODEL_DIR / "htdemucs_v4.xml"
MODEL_BIN = MODEL_DIR / "htdemucs_v4.bin"
HF_REPO = "Intel/demucs-openvino"
# Files on Hugging Face (same IR; Audacity renames to htdemucs_v4.* in the installer).
_HF_XML = "htdemucs_fwd.xml"
_HF_BIN = "htdemucs_fwd.bin"
CACHE_DIR = MODELS_DIR / "demucs-openvino" / "openvino-cache"


def model_ready() -> bool:
    return MODEL_XML.is_file() and MODEL_BIN.is_file()


def _install_audacity_names() -> None:
    """Copy HF filenames to htdemucs_v4.xml/.bin as used by the Audacity plugin."""
    src_xml = MODEL_DIR / _HF_XML
    src_bin = MODEL_DIR / _HF_BIN
    if src_xml.is_file() and not MODEL_XML.is_file():
        shutil.copy2(src_xml, MODEL_XML)
    if src_bin.is_file() and not MODEL_BIN.is_file():
        shutil.copy2(src_bin, MODEL_BIN)


def ensure_model(*, force: bool = False) -> Path:
    """Resolve HTDemucs v4 IR (models/ or anthill; optional Intel upstream)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if model_ready() and not force:
        return MODEL_XML

    ensure_anthill_files(
        [
            "demucs-openvino/htdemucs_v4/htdemucs_v4.xml",
            "demucs-openvino/htdemucs_v4/htdemucs_v4.bin",
        ],
        label="$music_separation",
        force=force,
    )
    _install_audacity_names()
    if model_ready():
        return MODEL_XML

    if upstream_fallback_enabled():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "$music_separation needs huggingface-hub to download the model: "
                "uv pip install huggingface-hub"
            ) from exc

        root = MODELS_DIR / "demucs-openvino"
        root.mkdir(parents=True, exist_ok=True)
        for name in (_HF_XML, _HF_BIN):
            hf_hub_download(
                HF_REPO,
                f"{MODEL_ID}/{name}",
                local_dir=str(root),
            )
        _install_audacity_names()

    if not model_ready():
        raise FileNotFoundError(
            f"Model not found under {MODEL_DIR}. "
            f"Run: uv run python tools/download_models.py"
        )
    return MODEL_XML


def configure_models_environment() -> Path:
    """Ensure models/ exists (mirrors music external)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODELS_PATH", str(MODELS_DIR))
    os.environ.setdefault("OPENVINO_CACHE_DIR", str(CACHE_DIR))
    return MODELS_DIR
