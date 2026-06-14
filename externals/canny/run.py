"""$canny — render Canny edge maps from images."""

from __future__ import annotations

import os

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.face_lib.cv2_io import require_cv2
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$canny turns each input image into a Canny edge map (white edges on black).

Example:
  @edges: $folder('photos') -> $canny()
  @fine: $file('scene.png') -> $canny(low=50, high=150, detect_resolution=768)

Optional args:
  low=100               Canny low threshold (0–255)
  high=200              Canny high threshold (0–255)
  detect_resolution=512 internal detection size (long side)

Setup: .venvs/media (controlnet-aux + opencv). No model download required.
AH_EMULATE_CANNY=1 for stub PNG output without deps.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_CANNY", "").lower() in ("1", "true", "yes")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    return int(raw)


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    cv2 = require_cv2()
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("$canny: failed to encode PNG")
    return buf.tobytes()


def _stub_canny_bgr(width: int, height: int) -> np.ndarray:
    cv2 = require_cv2()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    margin = max(8, min(width, height) // 8)
    cv2.rectangle(
        canvas,
        (margin, margin),
        (width - margin, height - margin),
        (255, 255, 255),
        2,
    )
    cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (255, 255, 255), 1)
    return canvas


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    if not images:
        raise RuntimeError(_HELP.strip())

    low_threshold = _int_arg(inp.args, "low", 100)
    high_threshold = _int_arg(inp.args, "high", 200)
    detect_resolution = _int_arg(inp.args, "detect_resolution", 512)

    new_images: list[str] = []
    for link in images:
        src = ctx.resolve_link_path(link)
        if not src.is_file():
            raise FileNotFoundError(f"$canny: image not found: {src}")

        cv2 = require_cv2()
        image_bgr = cv2.imread(str(src))
        if image_bgr is None:
            raise RuntimeError(f"$canny: unable to read image: {src}")

        if _emulate_enabled():
            h, w = image_bgr.shape[:2]
            edges = _stub_canny_bgr(w, h)
            new_images.append(ctx.new_link("images", ".png", _png_bytes(edges)))
            continue

        from externals.canny.pipeline import image_to_canny_bgr

        edges = image_to_canny_bgr(
            image_bgr,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            detect_resolution=detect_resolution,
        )
        new_images.append(ctx.new_link("images", ".png", _png_bytes(edges)))

    out.images = new_images
    return out
