"""Warm-worker launch command for $image2image."""

from __future__ import annotations

import sys


def build_image2image_worker_cmd() -> list[str]:
    """Launch warm worker in Anthill .venvs/media (comfy-kitchen for FP8)."""
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
            "$image2image: WARNING comfy_kitchen not installed — FP8 UNet will be slow. "
            "Run: UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media",
            flush=True,
        )
