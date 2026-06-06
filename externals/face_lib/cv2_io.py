"""OpenCV helpers for face externals."""

from __future__ import annotations


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "$face / $face_enhancer need OpenCV.\n"
            "  powershell -File tools\\setup_external_venvs.ps1\n"
            "Then set in .env:\n"
            "  AH_EXTERNAL_VENV_face=.venvs/media\n"
            "  AH_EXTERNAL_VENV_face_enhancer=.venvs/media"
        ) from exc
    return cv2
