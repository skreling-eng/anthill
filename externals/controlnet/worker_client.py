"""$controlnet uses per-job subprocess (no warm worker yet)."""

from __future__ import annotations


def worker_enabled() -> bool:
    return False


def run_via_worker(ctx, inp):
    raise RuntimeError("$controlnet warm worker is not implemented")


def terminate_worker() -> None:
    pass
