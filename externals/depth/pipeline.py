"""Depth maps via controlnet_aux MiDaS."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from externals.depth.model_paths import ensure_model

_LOCK = threading.Lock()
_DETECTOR: Any | None = None
_DETECTOR_GPU: bool | None = None


def _missing_controlnet_aux_error(exc: ImportError) -> RuntimeError:
    import sys
    from pathlib import Path

    from externals.invoke import subprocess_enabled, venv_python

    lines = [
        "$depth needs controlnet-aux in .venvs/media:",
        "  UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media",
        "  or: tools\\setup_external_venvs.ps1",
    ]
    expected = venv_python("depth")
    if expected:
        lines.append(f"  expected: {expected}")
        lines.append(f"  running:  {sys.executable}")
        if Path(sys.executable).resolve() != Path(expected).resolve():
            lines.append(
                "  $depth is running in the wrong Python. "
                "Ensure .env has AH_EXTERNAL_VENV_depth=.venvs/media "
                "and avoid AH_EXTERNAL_SUBPROCESS=0."
            )
    if not subprocess_enabled("depth"):
        lines.append(
            "  AH_EXTERNAL_SUBPROCESS=0 forces in-process mode in the base .venv "
            "(no controlnet-aux). Unset it or exclude depth from in-process mode."
        )
    return RuntimeError("\n".join(lines))


def _load_detector(*, use_gpu: bool) -> Any:
    global _DETECTOR, _DETECTOR_GPU
    with _LOCK:
        if _DETECTOR is not None and _DETECTOR_GPU == use_gpu:
            return _DETECTOR

        try:
            from controlnet_aux import MidasDetector
        except ImportError as exc:
            raise _missing_controlnet_aux_error(exc) from exc

        model_dir = ensure_model()
        detector = MidasDetector.from_pretrained(str(model_dir))
        if use_gpu:
            import torch

            if torch.cuda.is_available():
                detector.to(torch.device("cuda"))
        _DETECTOR = detector
        _DETECTOR_GPU = use_gpu
        return detector


def image_to_depth_bgr(
    image_bgr: np.ndarray,
    *,
    detect_resolution: int = 512,
    use_gpu: bool = True,
) -> np.ndarray:
    """Return a grayscale depth map (BGR, near=white = close, dark = far)."""
    detector = _load_detector(use_gpu=use_gpu)
    image_rgb = image_bgr[:, :, ::-1]
    long_side = max(image_bgr.shape[0], image_bgr.shape[1])
    depth_rgb = detector(
        image_rgb,
        detect_resolution=detect_resolution,
        image_resolution=long_side,
        output_type="np",
    )
    return depth_rgb[:, :, ::-1]
