"""Headless progress lines for long $avatar sampling runs."""

from __future__ import annotations

import functools
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_phase_lock = threading.Lock()
_current_phase = "working"
_phase_started = time.monotonic()


def set_avatar_phase(phase: str) -> None:
    global _current_phase, _phase_started
    with _phase_lock:
        if phase != _current_phase:
            _current_phase = phase
            _phase_started = time.monotonic()


def set_sampling_progress(step: int, total: int) -> None:
    """Update heartbeat while a single denoise step runs (often 30–120s each)."""
    set_avatar_phase(f"sampling step {step}/{total}")


def _avatar_print(msg: str) -> None:
    print(msg, flush=True)


def _stage_label(name: str, *, detail: str = "") -> str:
    return f"{name} ({detail})" if detail else name


def log_avatar_stage_start(name: str, *, detail: str = "") -> float:
    """Log ``$avatar: <stage>…`` and return a monotonic timestamp for pairing."""
    label = _stage_label(name, detail=detail)
    set_avatar_phase(label)
    _avatar_print(f"$avatar: {label}…")
    return time.perf_counter()


def log_avatar_stage_end(
    name: str,
    t0: float,
    *,
    detail: str = "",
    note: str = "",
) -> None:
    """Log ``$avatar: <stage> done (Xs)`` for a stage opened with :func:`log_avatar_stage_start`."""
    elapsed = time.perf_counter() - t0
    label = _stage_label(name, detail=detail)
    suffix = f", {note}" if note else ""
    _avatar_print(f"$avatar: {label} done ({elapsed:.1f}s{suffix})")


@contextmanager
def avatar_stage(name: str, *, detail: str = "") -> Iterator[None]:
    """Context manager that logs paired start/done lines for one stage."""
    t0 = log_avatar_stage_start(name, detail=detail)
    try:
        yield
    finally:
        log_avatar_stage_end(name, t0, detail=detail)


@contextmanager
def avatar_work_heartbeat(*, interval_s: float = 20.0) -> Iterator[None]:
    """Emit ``$avatar: still …`` while a long node runs."""
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_s):
            with _phase_lock:
                phase = _current_phase
                phase_elapsed = int(time.monotonic() - _phase_started)
            _avatar_print(f"$avatar: still {phase} ({phase_elapsed}s elapsed)…")

    thread = threading.Thread(target=_loop, name="avatar-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _patch_tqdm() -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        return
    if getattr(tqdm, "_anthill_avatar_patched", False):
        return

    _orig_init = tqdm.__init__
    _orig_update = tqdm.update
    _orig_close = tqdm.close
    _last_line = {"desc": "", "time": 0.0}
    _starts: dict[int, tuple[str, float]] = {}

    _LOG_SUBSTRINGS = (
        "sampling audio",
        "loading transformer",
        "wanvae encoding",
        "wanvae encoding frames",
        "initializing block swap",
        "initializing vace block swap",
    )

    def _stage_name_for_desc(desc: str) -> str | None:
        d = desc.lower()
        if "sampling audio" in d:
            return "sampling"
        if "loading transformer" in d:
            return "load transformer to GPU"
        if "wanvae encoding frames" in d:
            return "WanVAE encode frames"
        if "wanvae" in d:
            return "WanVAE encode"
        if "initializing block swap" in d or "initializing vace block swap" in d:
            return "block swap init"
        return None

    def _should_log(desc: str) -> bool:
        d = desc.lower()
        return any(s in d for s in _LOG_SUBSTRINGS)

    def _maybe_log_progress(desc: str, n: int, total: int | None) -> None:
        if not _should_log(desc):
            return
        now = time.monotonic()
        interval = 10.0 if "sampling audio" in desc.lower() else 20.0
        if (
            desc == _last_line["desc"]
            and now - _last_line["time"] < interval
            and total is not None
            and n < total
        ):
            return
        total_s = str(total) if total is not None else "?"
        stage = _stage_name_for_desc(desc)
        label = stage or desc
        _avatar_print(f"$avatar: {label} {n}/{total_s}")
        _last_line["desc"] = desc
        _last_line["time"] = now
        if stage is not None:
            set_avatar_phase(f"{stage} {n}/{total_s}")

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.5)
        _orig_init(self, *args, **kwargs)
        desc = str(getattr(self, "desc", "") or "")
        if not _should_log(desc):
            return
        stage = _stage_name_for_desc(desc)
        if stage is None:
            return
        total = getattr(self, "total", None)
        detail = f"{total} items" if total is not None else ""
        _starts[id(self)] = (stage, log_avatar_stage_start(stage, detail=detail))

    def _patched_update(self, n=1):
        result = _orig_update(self, n)
        _maybe_log_progress(
            str(getattr(self, "desc", "") or ""),
            int(getattr(self, "n", 0)),
            getattr(self, "total", None),
        )
        return result

    def _patched_close(self, *args, **kwargs):
        desc = str(getattr(self, "desc", "") or "")
        started = _starts.pop(id(self), None)
        if started is not None:
            stage, t0 = started
            log_avatar_stage_end(stage, t0)
        elif _should_log(desc):
            _maybe_log_progress(
                desc,
                int(getattr(self, "n", 0)),
                getattr(self, "total", None),
            )
        return _orig_close(self, *args, **kwargs)

    tqdm.__init__ = _patched_init  # type: ignore[method-assign]
    tqdm.update = _patched_update  # type: ignore[method-assign]
    tqdm.close = _patched_close  # type: ignore[method-assign]
    tqdm._anthill_avatar_patched = True  # type: ignore[attr-defined]


def _find_loaded_module(*suffixes: str):
    import sys

    for name, mod in sys.modules.items():
        if any(name.endswith(s) or s in name for s in suffixes):
            yield mod


def _patch_multitalk_loop() -> None:
    for mod in _find_loaded_module("multitalk_loop"):
        orig = getattr(mod, "multitalk_loop", None)
        if orig is None or not callable(orig) or getattr(orig, "_anthill_avatar_patched", False):
            continue

        def wrapped(*args: Any, _orig: Callable = orig, **kwargs: Any):
            with avatar_stage("multitalk loop"):
                return _orig(*args, **kwargs)

        wrapped._anthill_avatar_patched = True  # type: ignore[attr-defined]
        mod.multitalk_loop = wrapped
        return


def _patch_wan_vae_encode() -> None:
    for mod in _find_loaded_module("wan_video_vae"):
        cls = getattr(mod, "WanVideoVAE", None)
        if cls is None:
            continue
        orig = cls.encode
        if getattr(orig, "_anthill_avatar_patched", False):
            return

        def encode(
            self,
            videos,
            device,
            tiled=False,
            end_=False,
            tile_size=None,
            tile_stride=None,
            pbar=True,
            sample=False,
            _orig=orig,
        ):  # noqa: N802
            n = len(videos) if hasattr(videos, "__len__") else 1
            mode = "tiled" if tiled else "full"
            detail = f"{mode}, {n} clip(s)"
            with avatar_stage("WanVAE encode", detail=detail):
                return _orig(
                    self,
                    videos,
                    device,
                    tiled=tiled,
                    end_=end_,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    pbar=pbar,
                    sample=sample,
                )

        encode._anthill_avatar_patched = True  # type: ignore[attr-defined]
        cls.encode = encode  # type: ignore[method-assign]
        return


def _patch_park_dit_for_vae_encode() -> None:
    try:
        import externals.comfy_inprocess.wan_dit_vram as wd
    except ImportError:
        return
    orig = getattr(wd, "park_dit_for_vae_encode", None)
    if orig is None or getattr(orig, "_anthill_avatar_patched", False):
        return

    def park_dit_for_vae_encode(patcher, transformer, _orig=orig):
        t0 = log_avatar_stage_start("park DiT for VAE encode")
        before_mib, after_mib = _orig(patcher, transformer)
        log_avatar_stage_end(
            "park DiT for VAE encode",
            t0,
            note=f"GPU {before_mib:.0f} -> {after_mib:.0f} MiB",
        )
        return before_mib, after_mib

    park_dit_for_vae_encode._anthill_avatar_patched = True  # type: ignore[attr-defined]
    wd.park_dit_for_vae_encode = park_dit_for_vae_encode


def install_avatar_progress_logging() -> None:
    """Install headless progress hooks for WanVideoWrapper avatar runs."""
    _patch_tqdm()
    _patch_wan_vae_encode()
    _patch_multitalk_loop()
    _patch_park_dit_for_vae_encode()
