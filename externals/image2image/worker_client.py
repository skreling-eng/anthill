"""Warm subprocess pool for $image2image (avoids reloading ~28GB each job)."""

from __future__ import annotations

from externals.comfy_inprocess.warm_worker import WarmWorkerConfig, WarmWorkerPool
from externals.image2image.worker_cmd import build_image2image_worker_cmd

_CONFIG = WarmWorkerConfig(name="image2image", worker_module="externals.image2image.worker")
_POOL = WarmWorkerPool(_CONFIG, build_cmd=build_image2image_worker_cmd)


def worker_enabled() -> bool:
    return _POOL.enabled()


def run_via_worker(ctx, inp):
    return _POOL.run(ctx, inp)


def terminate_worker() -> None:
    _POOL.terminate()
