"""$face_enhancer — x4 neural face enhancement on aligned face crops."""

from __future__ import annotations

import os

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.face_lib.cv2_io import require_cv2
from externals.face_lib.pipeline import enhance_face_bgr
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$face_enhancer runs the DeepFaceLab FaceEnhancer model on each input face image.

Example:
  @faces: $folder('faces') -> $face_enhancer

Optional args:
  preserve_size=1   keep input resolution (default)
  is_tanh=0         input already in [-1, 1]

Models: models/face/FaceEnhancer.npy
Setup: powershell -File tools\\copy_face_models.ps1
Uses PyTorch in .venvs/media. GPU when available (AH_FACE_GPU=1).
"""


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_FACE_ENHANCER", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(args: dict[str, str], key: str, default: bool = True) -> bool:
    raw = args.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    cv2 = require_cv2()

    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("$face_enhancer: failed to encode PNG")
    return buf.tobytes()


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    if not images:
        raise RuntimeError(_HELP.strip())

    preserve_size = _truthy(inp.args, "preserve_size", True)
    is_tanh = _truthy(inp.args, "is_tanh", False)
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
            raise FileNotFoundError(f"$face_enhancer: image not found: {src}")

        if _emulate_enabled():
            new_images.append(ctx.new_link("images", ".png", src.read_bytes()))
            continue

        cv2 = require_cv2()
        image_bgr = cv2.imread(str(src))
        if image_bgr is None:
            raise RuntimeError(f"$face_enhancer: unable to read image: {src}")

        enhanced = enhance_face_bgr(
            image_bgr,
            preserve_size=preserve_size,
            is_tanh=is_tanh,
            place_models_on_cpu=cpu,
            run_on_cpu=cpu,
        )
        new_images.append(ctx.new_link("images", ".png", _png_bytes(enhanced)))

    out.images = new_images
    return out
