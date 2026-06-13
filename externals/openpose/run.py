"""$openpose — render OpenPose skeleton maps from images."""

from __future__ import annotations

import os

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.face_lib.cv2_io import require_cv2
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$openpose turns each input image into an OpenPose-style skeleton map.

Example:
  @poses: $folder('photos') -> $openpose()
  @full: $file('dancer.png') -> $openpose(hand=1, face=1, detect_resolution=768)

Optional args:
  detect_resolution=512   internal detection size (long side)
  hand=0                  include hand keypoints when 1
  face=0                  include face keypoints when 1

Models: models/openpose/body_pose_model.pth (+ hand/face weights when enabled)
Setup: uv run python tools/download_models.py --upstream-fallback
       .venvs/media (controlnet-aux + torch). AH_OPENPOSE_GPU=0 forces CPU.
AH_EMULATE_OPENPOSE=1 for stub PNG output without models.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_OPENPOSE", "").lower() in ("1", "true", "yes")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    return int(raw)


def _truthy(args: dict[str, str], key: str) -> bool:
    return args.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    cv2 = require_cv2()
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("$openpose: failed to encode PNG")
    return buf.tobytes()


def _stub_skeleton_bgr(width: int, height: int) -> np.ndarray:
    cv2 = require_cv2()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    color = (0, 255, 85)
    cv2.line(canvas, (cx, cy - height // 4), (cx, cy + height // 4), color, 4)
    cv2.line(canvas, (cx, cy - height // 6), (cx - width // 5, cy), color, 3)
    cv2.line(canvas, (cx, cy - height // 6), (cx + width // 5, cy), color, 3)
    cv2.line(canvas, (cx, cy + height // 6), (cx - width // 6, cy + height // 3), color, 3)
    cv2.line(canvas, (cx, cy + height // 6), (cx + width // 6, cy + height // 3), color, 3)
    cv2.circle(canvas, (cx, cy - height // 3), max(8, width // 20), color, -1)
    return canvas


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    if not images:
        raise RuntimeError(_HELP.strip())

    detect_resolution = _int_arg(inp.args, "detect_resolution", 512)
    include_hand = _truthy(inp.args, "hand")
    include_face = _truthy(inp.args, "face")
    use_gpu = os.environ.get("AH_OPENPOSE_GPU", "1").strip().lower() not in (
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
            raise FileNotFoundError(f"$openpose: image not found: {src}")

        cv2 = require_cv2()
        image_bgr = cv2.imread(str(src))
        if image_bgr is None:
            raise RuntimeError(f"$openpose: unable to read image: {src}")

        if _emulate_enabled():
            h, w = image_bgr.shape[:2]
            skeleton = _stub_skeleton_bgr(w, h)
            new_images.append(ctx.new_link("images", ".png", _png_bytes(skeleton)))
            continue

        from externals.openpose.pipeline import image_to_skeleton_bgr

        skeleton = image_to_skeleton_bgr(
            image_bgr,
            detect_resolution=detect_resolution,
            include_hand=include_hand,
            include_face=include_face,
            use_gpu=use_gpu,
        )
        new_images.append(ctx.new_link("images", ".png", _png_bytes(skeleton)))

    out.images = new_images
    return out
