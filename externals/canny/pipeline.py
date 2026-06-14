"""Canny edge maps via controlnet_aux."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

_LOCK = threading.Lock()
_DETECTOR: Any | None = None


def _missing_controlnet_aux_error(exc: ImportError) -> RuntimeError:
    import sys
    from pathlib import Path

    from externals.invoke import subprocess_enabled, venv_python

    lines = [
        "$canny needs controlnet-aux in .venvs/media:",
        "  UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media",
        "  or: tools\\setup_external_venvs.ps1",
    ]
    expected = venv_python("canny")
    if expected:
        lines.append(f"  expected: {expected}")
        lines.append(f"  running:  {sys.executable}")
        if Path(sys.executable).resolve() != Path(expected).resolve():
            lines.append(
                "  $canny is running in the wrong Python. "
                "Ensure .env has AH_EXTERNAL_VENV_canny=.venvs/media "
                "and avoid AH_EXTERNAL_SUBPROCESS=0."
            )
    if not subprocess_enabled("canny"):
        lines.append(
            "  AH_EXTERNAL_SUBPROCESS=0 forces in-process mode in the base .venv "
            "(no controlnet-aux). Unset it or exclude canny from in-process mode."
        )
    return RuntimeError("\n".join(lines))


def _load_detector() -> Any:
    global _DETECTOR
    with _LOCK:
        if _DETECTOR is not None:
            return _DETECTOR
        try:
            from controlnet_aux import CannyDetector
        except ImportError as exc:
            raise _missing_controlnet_aux_error(exc) from exc
        _DETECTOR = CannyDetector()
        return _DETECTOR


def image_to_canny_bgr(
    image_bgr: np.ndarray,
    *,
    low_threshold: int = 100,
    high_threshold: int = 200,
    detect_resolution: int = 512,
) -> np.ndarray:
    """Return a Canny edge map (BGR, white edges on black background)."""
    detector = _load_detector()
    image_rgb = image_bgr[:, :, ::-1]
    long_side = max(image_bgr.shape[0], image_bgr.shape[1])
    edges_rgb = detector(
        image_rgb,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        detect_resolution=detect_resolution,
        image_resolution=long_side,
        output_type="np",
    )
    return edges_rgb[:, :, ::-1]
