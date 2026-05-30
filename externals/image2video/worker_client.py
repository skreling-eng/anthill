"""Warm subprocess pool for $image2video (avoids reloading ~22GB each job)."""

from __future__ import annotations

from externals.comfy_inprocess.warm_worker import WarmWorkerConfig, WarmWorkerPool

_CONFIG = WarmWorkerConfig(name="image2video", worker_module="externals.image2video.worker")
_POOL = WarmWorkerPool(_CONFIG)


def worker_enabled() -> bool:
    return _POOL.enabled()


def run_via_worker(ctx, inp):
    return _POOL.run(ctx, inp)


def terminate_worker() -> None:
    _POOL.terminate()
