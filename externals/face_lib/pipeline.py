"""Face detect / align / enhance pipeline."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from externals.face_lib.face_type import FaceType, face_type_from_string
from externals.face_lib import landmarks as LandmarksProcessor
from externals.face_lib.model_paths import align_ready, enhancer_ready
from externals.face_lib.pytorch_align import extract_landmarks

_LOCK = threading.Lock()
_ENHANCER: Any | None = None


def _enhancer(*, use_cpu: bool = False):
    global _ENHANCER
    with _LOCK:
        if _ENHANCER is None:
            from externals.face_lib.face_enhancer_pt import FaceEnhancer

            _ENHANCER = FaceEnhancer(use_cpu=use_cpu)
        return _ENHANCER


def extract_face_bgr(
    image_bgr: np.ndarray,
    *,
    size: int = 256,
    face_type: FaceType = FaceType.FULL,
    face_index: int = 0,
    landmarks_3d: bool = False,
    place_models_on_cpu: bool = False,
) -> np.ndarray | None:
    if not align_ready(landmarks_3d=landmarks_3d):
        ensure_align_models(landmarks_3d=landmarks_3d)
    if not align_ready(landmarks_3d=landmarks_3d):
        raise FileNotFoundError(
            "$face models missing under models/face/ "
            "(need s3fd-619a316812.pth and 2DFAN4-11f355bf06.pth.tar). "
            "Run: uv run python tools/download_models.py"
        )

    landmarks = extract_landmarks(
        image_bgr,
        landmarks_3d=landmarks_3d,
        use_cpu=place_models_on_cpu,
        face_index=face_index,
    )
    if landmarks is None:
        return None

    mat = LandmarksProcessor.get_transform_mat(
        landmarks,
        size,
        face_type,
    )
    import cv2

    return cv2.warpAffine(image_bgr, mat, (size, size), flags=cv2.INTER_CUBIC)


def enhance_face_bgr(
    face_bgr: np.ndarray,
    *,
    preserve_size: bool = True,
    is_tanh: bool = False,
    place_models_on_cpu: bool = False,
    run_on_cpu: bool = False,
) -> np.ndarray:
    if not enhancer_ready():
        from externals.face_lib.model_paths import ensure_enhancer_model

        ensure_enhancer_model()
    if not enhancer_ready():
        raise FileNotFoundError(
            "FaceEnhancer.npy missing under models/face/. "
            "Run: uv run python tools/download_models.py"
        )

    use_cpu = place_models_on_cpu or run_on_cpu
    enhancer = _enhancer(use_cpu=use_cpu)
    rgb = face_bgr[:, :, ::-1].astype(np.float32) / 255.0
    enhanced = enhancer.enhance(rgb, is_tanh=is_tanh, preserve_size=preserve_size)
    out = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return out[:, :, ::-1]


def parse_face_type(raw: str) -> FaceType:
    return face_type_from_string(raw)
