"""OpenPose skeleton map rendering via controlnet_aux."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from externals.openpose.model_paths import BODY_PTH, FACE_PTH, HAND_PTH, ensure_model

_LOCK = threading.Lock()
_DETECTOR: Any | None = None
_DETECTOR_KEY: tuple[bool, bool, bool] | None = None


def _missing_controlnet_aux_error(exc: ImportError) -> RuntimeError:
    import sys
    from pathlib import Path

    from externals.invoke import subprocess_enabled, venv_python

    lines = [
        "$openpose needs controlnet-aux in .venvs/media:",
        "  UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media",
        "  or: tools\\setup_external_venvs.ps1",
    ]
    expected = venv_python("openpose")
    if expected:
        lines.append(f"  expected: {expected}")
        lines.append(f"  running:  {sys.executable}")
        if Path(sys.executable).resolve() != Path(expected).resolve():
            lines.append(
                "  $openpose is running in the wrong Python. "
                "Ensure .env has AH_EXTERNAL_VENV_openpose=.venvs/media "
                "and avoid AH_EXTERNAL_SUBPROCESS=0."
            )
    if not subprocess_enabled("openpose"):
        lines.append(
            "  AH_EXTERNAL_SUBPROCESS=0 forces in-process mode in the base .venv "
            "(no controlnet-aux). Unset it or exclude openpose from in-process mode."
        )
    return RuntimeError("\n".join(lines))


def _require_controlnet_aux():
    try:
        from controlnet_aux import OpenposeDetector
        from controlnet_aux.open_pose.body import Body
        from controlnet_aux.open_pose.face import Face
        from controlnet_aux.open_pose.hand import Hand
    except ImportError as exc:
        raise _missing_controlnet_aux_error(exc) from exc
    return OpenposeDetector, Body, Hand, Face


def _load_detector(
    *,
    include_hand: bool,
    include_face: bool,
    use_gpu: bool,
) -> Any:
    global _DETECTOR, _DETECTOR_KEY
    key = (include_hand, include_face, use_gpu)
    with _LOCK:
        if _DETECTOR is not None and _DETECTOR_KEY == key:
            return _DETECTOR

        OpenposeDetector, Body, Hand, Face = _require_controlnet_aux()
        model_dir = ensure_model(include_hand=include_hand, include_face=include_face)
        body = Body(str(model_dir / BODY_PTH))
        hand = Hand(str(model_dir / HAND_PTH)) if include_hand else None
        face = Face(str(model_dir / FACE_PTH)) if include_face else None
        detector = OpenposeDetector(body, hand, face)
        if use_gpu:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda")
                detector.body_estimation.to(device)
                if detector.hand_estimation is not None:
                    detector.hand_estimation.to(device)
                if detector.face_estimation is not None:
                    detector.face_estimation.to(device)
        _DETECTOR = detector
        _DETECTOR_KEY = key
        return detector


def image_to_skeleton_bgr(
    image_bgr: np.ndarray,
    *,
    detect_resolution: int = 512,
    include_hand: bool = False,
    include_face: bool = False,
    use_gpu: bool = True,
) -> np.ndarray:
    """Return an OpenPose-style skeleton map (BGR, black background)."""
    detector = _load_detector(
        include_hand=include_hand,
        include_face=include_face,
        use_gpu=use_gpu,
    )
    image_rgb = image_bgr[:, :, ::-1]
    skeleton_rgb = detector(
        image_rgb,
        detect_resolution=detect_resolution,
        image_resolution=max(image_bgr.shape[0], image_bgr.shape[1]),
        include_body=True,
        include_hand=include_hand,
        include_face=include_face,
        output_type="np",
    )
    return skeleton_rgb[:, :, ::-1]
