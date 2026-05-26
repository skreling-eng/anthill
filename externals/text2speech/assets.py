"""Download Kokoro assets into models/kokoro (no HuggingFace hub cache)."""

from __future__ import annotations

import os
from pathlib import Path

from externals.text2speech.model_paths import (
    DEFAULT_MODEL_FILE,
    kokoro_root,
)

# Pip kokoro checkpoints per repo (see kokoro.model.KModel.MODEL_NAMES).
_PIPELINE_WEIGHTS: dict[str, str] = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}

# v0.19 HF clone weights (keys like "net") — legacy backend only, not pip KModel.
_LEGACY_WEIGHTS = DEFAULT_MODEL_FILE


def checkpoint_format(path: Path) -> str:
    """Return 'v1' (pip KModel), 'v0' (legacy build_model), or 'unknown'."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        return "unknown"
    keys = set(data.keys())
    if "net" in keys:
        return "v0"
    if keys & {"bert", "decoder", "text_encoder", "predictor"}:
        return "v1"
    return "unknown"


def _pipeline_weights_name(repo_id: str) -> str:
    name = _PIPELINE_WEIGHTS.get(repo_id)
    if not name:
        raise ValueError(
            f"$text2speech: unknown repo_id {repo_id!r}. "
            f"Known: {', '.join(_PIPELINE_WEIGHTS)}"
        )
    return name


def _hub_env() -> None:
    """Keep hub traffic out of the user cache; files land under models/kokoro only."""
    root = kokoro_root()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Redirect any incidental hub cache into the model tree (not ~/.cache).
    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / ".hub_tmp")
    os.environ["HF_HUB_CACHE"] = str(root / ".hub_tmp")


def _download(repo_id: str, filename: str, root: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    _hub_env()
    base = (root or kokoro_root()).resolve()
    base.mkdir(parents=True, exist_ok=True)
    print(f"$text2speech: downloading {filename} -> {base}", flush=True)
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(base),
    )
    path = Path(out).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"$text2speech: download failed: {path}")
    return path


def weights_filename(repo_id: str, root: Path | None = None) -> str:
    """Pipeline checkpoint filename (always v1.x, never kokoro-v0_19.pth)."""
    return _pipeline_weights_name(repo_id)


def has_pipeline_weights(repo_id: str, root: Path | None = None) -> bool:
    base = root or kokoro_root()
    path = base / _pipeline_weights_name(repo_id)
    return path.is_file() and checkpoint_format(path) == "v1"


def has_legacy_weights(root: Path | None = None) -> bool:
    base = root or kokoro_root()
    path = base / _LEGACY_WEIGHTS
    return path.is_file() and checkpoint_format(path) == "v0"


def ensure_config(repo_id: str, root: Path | None = None) -> Path:
    base = root or kokoro_root()
    path = base / "config.json"
    if path.is_file():
        return path.resolve()
    return _download(repo_id, "config.json", base)


def ensure_weights(repo_id: str, root: Path | None = None) -> Path:
    """Load or download pip-kokoro v1 checkpoint (not v0.19)."""
    base = root or kokoro_root()
    name = _pipeline_weights_name(repo_id)
    path = base / name
    if path.is_file():
        fmt = checkpoint_format(path)
        if fmt == "v1":
            return path.resolve()
        raise RuntimeError(
            f"$text2speech: {path.name} is not a pip-kokoro checkpoint (found {fmt!r} layout). "
            f"Use AH_TEXT2SPEECH_BACKEND=legacy or replace with {name} from {repo_id}."
        )
    v0 = base / _LEGACY_WEIGHTS
    if v0.is_file() and checkpoint_format(v0) == "v0":
        print(
            f"$text2speech: {_LEGACY_WEIGHTS} is v0.19 (legacy); "
            f"downloading {name} for pipeline backend",
            flush=True,
        )
    return _download(repo_id, name, base)


def ensure_model_assets(
    repo_id: str = "hexgrad/Kokoro-82M",
    root: Path | None = None,
) -> tuple[Path, Path]:
    """config.json + checkpoint under models/kokoro."""
    config = ensure_config(repo_id, root)
    weights = ensure_weights(repo_id, root)
    return config, weights


def ensure_voice_pack(
    voice: str,
    *,
    repo_id: str = "hexgrad/Kokoro-82M",
    root: Path | None = None,
) -> Path:
    """voices/<name>.pt under models/kokoro."""
    base = root or kokoro_root()
    name = Path(voice.strip()).stem or voice.strip()
    if not name:
        raise ValueError("$text2speech: empty voice name")
    rel = f"voices/{name}.pt"
    path = base / rel
    if path.is_file():
        return path.resolve()
    return _download(repo_id, rel, base)
