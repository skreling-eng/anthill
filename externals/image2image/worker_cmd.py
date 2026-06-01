"""Warm-worker launch command for $image2image."""

from __future__ import annotations

import os
import sys


def build_image2image_worker_cmd() -> list[str]:
    """Prefer ComfyUI venv (comfy_kitchen + comfy_aimdo) over .venvs/media for sampling speed."""
    force_media = os.environ.get("AH_IMAGE2IMAGE_USE_MEDIA_VENV", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not force_media:
        from externals.comfy_inprocess.bootstrap import resolve_comfy_python

        comfy_py = resolve_comfy_python()
        if comfy_py is not None:
            return [str(comfy_py), "-m", "externals.image2image.worker"]

    from externals.comfy_inprocess.warm_worker import WarmWorkerConfig, default_comfy_worker_cmd

    return default_comfy_worker_cmd(
        WarmWorkerConfig(name="image2image", worker_module="externals.image2image.worker")
    )


def log_worker_backend() -> None:
    """Log interpreter and optional fast-path backends (stderr in warm worker)."""
    print(f"$image2image: worker python {sys.executable}", flush=True)
    try:
        import comfy_kitchen  # noqa: F401

        print(
            "$image2image: comfy_kitchen package present "
            "(cuda FP8 status logged after comfy bootstrap)",
            flush=True,
        )
    except ImportError:
        print(
            "$image2image: WARNING comfy_kitchen not installed — UNet steps can be "
            "10–50× slower than ComfyUI. Set AH_COMFY_PYTHON to your ComfyUI "
            ".venv\\Scripts\\python.exe (or unset AH_EXTERNAL_VENV_image2image).",
            flush=True,
        )
