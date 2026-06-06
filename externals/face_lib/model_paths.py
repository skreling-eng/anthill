"""Local paths for face models under models/face/."""

from __future__ import annotations

from pathlib import Path

from externals.image.model_paths import models_roots

# PyTorch face-alignment weights (see tools/fetch_face_alignment_models.ps1).
S3FD_PTH = "s3fd-619a316812.pth"
FAN2D_PTH = "2DFAN4-11f355bf06.pth.tar"
FAN3D_PTH = "3DFAN4-7835d9f11d.pth.tar"
FAN_DEPTH_PTH = "depth-2a464da4ea.pth.tar"
FACE_ENHANCER_NPY = "FaceEnhancer.npy"

# URLs used by face-alignment when models are not bundled locally.
FAN2D_URL = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar"
FAN3D_URL = "https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar"
FAN_DEPTH_URL = "https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar"

ALIGN_FILES_2D = (S3FD_PTH, FAN2D_PTH)
ALIGN_FILES_3D = (S3FD_PTH, FAN3D_PTH, FAN_DEPTH_PTH)


def face_models_dir() -> Path:
    for root in models_roots():
        candidate = root / "face"
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / "face"


def model_path(name: str) -> Path:
    return face_models_dir() / name


def _files_present(names: tuple[str, ...]) -> bool:
    root = face_models_dir()
    return all((root / name).is_file() for name in names)


def align_ready(*, landmarks_3d: bool = False) -> bool:
    names = ALIGN_FILES_3D if landmarks_3d else ALIGN_FILES_2D
    return _files_present(names)


def models_ready(*, landmarks_3d: bool = False) -> bool:
    return align_ready(landmarks_3d=landmarks_3d)


def enhancer_ready() -> bool:
    return model_path(FACE_ENHANCER_NPY).is_file()


def ensure_align_models(*, landmarks_3d: bool = False) -> None:
    """Download missing $face weights from the anthill HF bundle."""
    if align_ready(landmarks_3d=landmarks_3d):
        return
    from externals.anthill_models import ensure_anthill_files

    rels = [f"face/{name}" for name in (ALIGN_FILES_3D if landmarks_3d else ALIGN_FILES_2D)]
    ensure_anthill_files(rels, label="$face")


def ensure_enhancer_model() -> None:
    if enhancer_ready():
        return
    from externals.anthill_models import ensure_anthill_file

    ensure_anthill_file(f"face/{FACE_ENHANCER_NPY}", label="$face_enhancer")
