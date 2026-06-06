"""$face — detect and crop aligned faces from images."""

from __future__ import annotations

import os

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.face_lib.cv2_io import require_cv2
from externals.face_lib.pipeline import extract_face_bgr, parse_face_type
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$face extracts an aligned face crop from each input image.

Example:
  @portraits: $folder('photos') -> $face(size=256, face_type='full_face')

Optional args:
  size=256          output width/height in pixels
  face_type=full_face   half_face, midfull_face, full_face, whole_face, head, ...
  face_index=0      which detected face (0 = largest)
  landmarks_3d=0    use 3D FAN model instead of 2D

Models: models/face/s3fd-619a316812.pth, 2DFAN4-11f355bf06.pth.tar
Setup: uv run python tools/download_models.py
AH_EMULATE_FACE=1 for stub PNG output without models.
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_FACE", "").lower() in ("1", "true", "yes")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    return int(raw)


def _truthy(args: dict[str, str], key: str) -> bool:
    return args.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    cv2 = require_cv2()

    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("$face: failed to encode PNG")
    return buf.tobytes()


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    if not images:
        raise RuntimeError(_HELP.strip())

    size = _int_arg(inp.args, "size", 256)
    face_index = _int_arg(inp.args, "face_index", 0)
    face_type = parse_face_type(inp.args.get("face_type", "full_face"))
    landmarks_3d = _truthy(inp.args, "landmarks_3d")
    cpu = os.environ.get("AH_FACE_GPU", "1").strip().lower() in (
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
            raise FileNotFoundError(f"$face: image not found: {src}")

        if _emulate_enabled():
            new_images.append(ctx.new_link("images", ".png", src.read_bytes()))
            continue

        cv2 = require_cv2()
        image_bgr = cv2.imread(str(src))
        if image_bgr is None:
            raise RuntimeError(f"$face: unable to read image: {src}")

        face = extract_face_bgr(
            image_bgr,
            size=size,
            face_type=face_type,
            face_index=face_index,
            landmarks_3d=landmarks_3d,
            place_models_on_cpu=cpu,
        )
        if face is None:
            raise RuntimeError(f"$face: no face detected in {src}")

        new_images.append(ctx.new_link("images", ".png", _png_bytes(face)))

    out.images = new_images
    return out
