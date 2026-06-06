"""PyTorch face detection + 68-point landmarks (face-alignment / S3FD + FAN)."""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from externals.face_lib.model_paths import (
    FAN2D_URL,
    FAN3D_URL,
    FAN_DEPTH_URL,
    S3FD_PTH,
    ensure_align_models,
    model_path,
)

_LOCK = threading.Lock()
_ALIGNER_2D: Any | None = None
_ALIGNER_3D: Any | None = None
_PATCHED = False


def _use_cpu(*, use_cpu: bool = False) -> bool:
    if use_cpu:
        return True
    raw = os.environ.get("AH_FACE_GPU", "1").strip().lower()
    return raw in ("0", "false", "no", "off", "cpu")


def _torch_device(*, use_cpu: bool = False) -> str:
    if _use_cpu(use_cpu=use_cpu):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _patch_face_alignment_loading() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import face_alignment.api as fa_api

    url_to_file = {
        FAN2D_URL: model_path("2DFAN4-11f355bf06.pth.tar"),
        FAN3D_URL: model_path("3DFAN4-7835d9f11d.pth.tar"),
        FAN_DEPTH_URL: model_path("depth-2a464da4ea.pth.tar"),
    }
    original = fa_api.load_file_from_url

    def load_file_from_url(url, *args, **kwargs):
        local = url_to_file.get(url)
        if local is not None and local.is_file():
            return str(local)
        return original(url, *args, **kwargs)

    fa_api.load_file_from_url = load_file_from_url
    _PATCHED = True


def _aligner(*, landmarks_3d: bool = False, use_cpu: bool = False):
    global _ALIGNER_2D, _ALIGNER_3D
    import face_alignment

    ensure_align_models(landmarks_3d=landmarks_3d)
    _patch_face_alignment_loading()

    device = _torch_device(use_cpu=use_cpu)
    s3fd = model_path(S3FD_PTH)
    if not s3fd.is_file():
        raise FileNotFoundError(
            f"$face model missing: {s3fd}\n"
            "Run: uv run python tools/download_models.py\n"
            "  or: powershell -File tools\\fetch_face_alignment_models.ps1"
        )

    detector_kwargs = {"path_to_detector": str(s3fd)}
    with _LOCK:
        if landmarks_3d:
            if _ALIGNER_3D is None:
                _ALIGNER_3D = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.THREE_D,
                    device=device,
                    flip_input=False,
                    face_detector="sfd",
                    face_detector_kwargs=detector_kwargs,
                    compile=False,
                )
            return _ALIGNER_3D
        if _ALIGNER_2D is None:
            _ALIGNER_2D = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                flip_input=False,
                face_detector="sfd",
                face_detector_kwargs=detector_kwargs,
                compile=False,
            )
        return _ALIGNER_2D


def extract_landmarks(
    image_bgr: np.ndarray,
    *,
    landmarks_3d: bool = False,
    use_cpu: bool = False,
    face_index: int = 0,
) -> np.ndarray | None:
    """Return 68x2 landmark coordinates for the selected face, or None."""
    aligner = _aligner(landmarks_3d=landmarks_3d, use_cpu=use_cpu)
    image_rgb = image_bgr[:, :, ::-1]

    landmarks, _scores, bboxes = aligner.get_landmarks_from_image(
        image_rgb,
        return_bboxes=True,
    )
    if not landmarks or not bboxes:
        return None

    order = sorted(
        range(len(bboxes)),
        key=lambda i: _bbox_area(np.asarray(bboxes[i], dtype=np.float32)),
        reverse=True,
    )
    if face_index < 0 or face_index >= len(order):
        raise IndexError(
            f"face_index={face_index} out of range ({len(order)} face(s) detected)"
        )

    pts = np.asarray(landmarks[order[face_index]], dtype=np.float32)
    if pts.ndim != 2:
        return None
    if pts.shape[1] > 2:
        pts = pts[:, :2]
    return pts
