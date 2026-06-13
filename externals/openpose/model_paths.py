"""OpenPose weights under models/openpose/ (lllyasviel/Annotators)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import ensure_anthill_files
from externals.image.model_paths import models_roots

UPSTREAM_REPO = "lllyasviel/Annotators"

BODY_PTH = "body_pose_model.pth"
HAND_PTH = "hand_pose_model.pth"
FACE_PTH = "facenet.pth"

ALL_FILES = (BODY_PTH, HAND_PTH, FACE_PTH)


def openpose_models_dir() -> Path:
    for root in models_roots():
        candidate = root / "openpose"
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / "openpose"


def model_path(name: str) -> Path:
    return openpose_models_dir() / name


def _file_ready(name: str) -> bool:
    return model_path(name).is_file()


def model_ready(*, include_hand: bool = False, include_face: bool = False) -> bool:
    if not _file_ready(BODY_PTH):
        return False
    if include_hand and not _file_ready(HAND_PTH):
        return False
    if include_face and not _file_ready(FACE_PTH):
        return False
    return True


def _needed_files(*, include_hand: bool, include_face: bool) -> tuple[str, ...]:
    names = [BODY_PTH]
    if include_hand:
        names.append(HAND_PTH)
    if include_face:
        names.append(FACE_PTH)
    return tuple(names)


def _anthill_rels(names: tuple[str, ...]) -> list[str]:
    return [f"openpose/{name}" for name in names]


def _download_upstream(names: tuple[str, ...]) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "$openpose needs huggingface-hub to download models: "
            "uv sync --extra media"
        ) from exc

    dest = openpose_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        if _file_ready(name):
            continue
        print(f"$openpose: downloading {UPSTREAM_REPO}/{name} -> {dest}", flush=True)
        cached = hf_hub_download(UPSTREAM_REPO, name)
        target = dest / name
        if not target.is_file():
            target.write_bytes(Path(cached).read_bytes())


def ensure_model(
    *,
    include_hand: bool = False,
    include_face: bool = False,
    force: bool = False,
) -> Path:
    """Resolve OpenPose weights; fetch from anthill bundle or upstream HF."""
    needed = _needed_files(include_hand=include_hand, include_face=include_face)
    if model_ready(include_hand=include_hand, include_face=include_face) and not force:
        return openpose_models_dir()

    try:
        ensure_anthill_files(_anthill_rels(needed), label="$openpose", force=force)
    except Exception:
        pass
    if model_ready(include_hand=include_hand, include_face=include_face):
        return openpose_models_dir()

    _download_upstream(needed)

    if not model_ready(include_hand=include_hand, include_face=include_face):
        missing = [n for n in needed if not _file_ready(n)]
        raise FileNotFoundError(
            f"$openpose models missing under {openpose_models_dir()}: {', '.join(missing)}. "
            "Run: uv run python tools/download_models.py --upstream-fallback"
        )
    return openpose_models_dir()
