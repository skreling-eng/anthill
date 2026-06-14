"""$depth — render MiDaS depth maps from images."""

from __future__ import annotations

import os

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.face_lib.cv2_io import require_cv2
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$depth turns each input image into a MiDaS depth map (grayscale).

Example:
  @depth_maps: $folder('photos') -> $depth()
  @fine: $file('scene.png') -> $depth(detect_resolution=768)

Optional args:
  detect_resolution=512   internal detection size (long side)

Models: models/depth/dpt_hybrid-midas-501f0c75.pt
Setup: uv run python tools/download_models.py --upstream-fallback
       .venvs/media (controlnet-aux + torch). AH_DEPTH_GPU=0 forces CPU.
AH_EMULATE_DEPTH=1 for stub PNG output without models.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_DEPTH", "").lower() in ("1", "true", "yes")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    return int(raw)


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    cv2 = require_cv2()
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("$depth: failed to encode PNG")
    return buf.tobytes()


def _stub_depth_bgr(width: int, height: int) -> np.ndarray:
    gradient = np.linspace(255, 32, width, dtype=np.uint8)
    gray = np.tile(gradient, (height, 1))
    return np.stack([gray, gray, gray], axis=-1)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    if not images:
        raise RuntimeError(_HELP.strip())

    detect_resolution = _int_arg(inp.args, "detect_resolution", 512)
    use_gpu = os.environ.get("AH_DEPTH_GPU", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "cpu",
    )

    new_images: list[str] = []
    for link in images:
        src = ctx.resolve_link_path(link)
        if not src.is_file():
            raise FileNotFoundError(f"$depth: image not found: {src}")

        cv2 = require_cv2()
        image_bgr = cv2.imread(str(src))
        if image_bgr is None:
            raise RuntimeError(f"$depth: unable to read image: {src}")

        if _emulate_enabled():
            h, w = image_bgr.shape[:2]
            depth = _stub_depth_bgr(w, h)
            new_images.append(ctx.new_link("images", ".png", _png_bytes(depth)))
            continue

        from externals.depth.pipeline import image_to_depth_bgr

        depth = image_to_depth_bgr(
            image_bgr,
            detect_resolution=detect_resolution,
            use_gpu=use_gpu,
        )
        new_images.append(ctx.new_link("images", ".png", _png_bytes(depth)))

    out.images = new_images
    return out
