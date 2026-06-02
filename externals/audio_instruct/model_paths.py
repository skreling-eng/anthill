"""Local paths and downloads for $audio_instruct (Qwen2-Audio 4-bit)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled
from externals.audio_instruct.model_list import AudioInstructModel, get_audio_instruct_model
from externals.image.model_paths import models_roots


def _resolve_profile(profile: AudioInstructModel | str | None) -> AudioInstructModel:
    if isinstance(profile, AudioInstructModel):
        return profile
    return get_audio_instruct_model(profile or "default")


def model_dir(profile: AudioInstructModel | str | None = None) -> Path:
    m = _resolve_profile(profile)
    for root in models_roots():
        candidate = root / m.subdir
        if (candidate / "config.json").is_file():
            return candidate
    return models_roots()[0] / m.subdir


def _has_weights(path: Path) -> bool:
    if (path / "model.safetensors").is_file():
        return True
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shards = {str(v) for v in weight_map.values()}
        return bool(shards) and all((path / name).is_file() for name in shards)
    return any(path.glob("model-*.safetensors")) or any(path.glob("*.safetensors"))


def model_ready(profile: AudioInstructModel | str | None = None) -> bool:
    path = model_dir(profile)
    return (path / "config.json").is_file() and _has_weights(path)


def _upstream_blocked() -> bool:
    raw = os.environ.get("AH_MODEL_UPSTREAM_FALLBACK", "").strip().lower()
    return raw in ("0", "false", "no", "off")


def _download_upstream(m: AudioInstructModel, path: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "$audio_instruct needs huggingface-hub to download the model: "
            "uv sync --extra media"
        ) from exc

    print(f"$audio_instruct: downloading {m.hf_repo} -> {path}", flush=True)
    snapshot_download(m.hf_repo, local_dir=str(path))


def ensure_model(profile: AudioInstructModel | str | None = None, *, force: bool = False) -> Path:
    """Resolve Qwen2-Audio 4-bit weights under models/qwen-audio/."""
    m = _resolve_profile(profile)
    path = model_dir(m)
    path.mkdir(parents=True, exist_ok=True)
    if model_ready(m) and not force:
        return path

    try:
        ensure_anthill_tree(
            m.subdir.as_posix(),
            ready=lambda: model_ready(m),
            label="$audio_instruct",
            force=force,
        )
    except FileNotFoundError:
        print(
            "$audio_instruct: not in skreling-eng/anthill bundle yet, "
            "will try Hugging Face upstream",
            flush=True,
        )

    if model_ready(m):
        return model_dir(m)

    if not _upstream_blocked():
        _download_upstream(m, path)
        if model_ready(m):
            return model_dir(m)

    if not model_ready(m):
        raise FileNotFoundError(
            f"Model not ready under {path}.\n"
            f"  uv run python tools/download_models.py --upstream-fallback\n"
            f"  or: hf auth login  then re-run $audio_instruct\n"
            f"  Set AH_MODEL_UPSTREAM_FALLBACK=0 to forbid Hugging Face download."
        )
    return path
