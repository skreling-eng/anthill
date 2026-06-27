"""Sampler timing fixes for $avatar (defer DiT GPU load until after WanVAE encode)."""

from __future__ import annotations

import functools
import sys
import threading
from typing import Any, Callable

_state = threading.local()
_WRAPPED: Callable[..., Any] | None = None
_BLOCKSWAP_WRAPPED: Callable[..., Any] | None = None

# Modules that do ``from ..nodes_model_loading import load_weights`` (local binding).
_REBIND_SUFFIXES = (
    "nodes_model_loading",
    "nodes_sampler",
    "multitalk_loop",
    "skyreels.nodes",
)
_BLOCKSWAP_REBIND_SUFFIXES = (
    "ComfyUI-WanVideoWrapper.utils",
    "nodes_sampler",
    "multitalk_loop",
    "skyreels.nodes",
)


def begin_avatar_job(*, defer_transformer_load: bool = True) -> None:
    """Call once per avatar job before ``execute_prompt``."""
    _state.defer_transformer_load = defer_transformer_load
    _state.deferred_load_skipped = False
    _state.deferred_blockswap_skipped = False


def end_avatar_job() -> None:
    for attr in (
        "defer_transformer_load",
        "deferred_load_skipped",
        "deferred_blockswap_skipped",
    ):
        if hasattr(_state, attr):
            delattr(_state, attr)


def defer_transformer_load_active() -> bool:
    """True while the current avatar job defers DiT GPU load until after VAE encode."""
    return bool(getattr(_state, "defer_transformer_load", False))


def _should_skip_initial_load() -> bool:
    return defer_transformer_load_active() and not bool(
        getattr(_state, "deferred_load_skipped", False)
    )


def _should_skip_initial_blockswap() -> bool:
    return defer_transformer_load_active() and not bool(
        getattr(_state, "deferred_blockswap_skipped", False)
    )


def _log_stage(name: str, *, detail: str = "") -> None:
    from externals.avatar.progress import log_avatar_stage_end, log_avatar_stage_start

    t0 = log_avatar_stage_start(name, detail=detail)
    log_avatar_stage_end(name, t0, detail=detail)


def _module_matches(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith("." + suffix) or suffix in name


def _iter_rebind_modules():
    for name, mod in sys.modules.items():
        if mod is None:
            continue
        if any(_module_matches(name, suffix) for suffix in _REBIND_SUFFIXES):
            yield mod


def _make_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(orig)
    def load_weights(*args: Any, **kwargs: Any):
        if _should_skip_initial_load():
            _state.deferred_load_skipped = True
            _log_stage("defer DiT load", detail="until after VAE encode")
            return None
        return orig(*args, **kwargs)

    load_weights._anthill_avatar_patched = True  # type: ignore[attr-defined]
    return load_weights


def _rebind_wrapped_load_weights(wrapped: Callable[..., Any]) -> None:
    for mod in _iter_rebind_modules():
        if hasattr(mod, "load_weights"):
            mod.load_weights = wrapped  # type: ignore[assignment]


def _patch_load_weights_defer() -> None:
    global _WRAPPED
    if _WRAPPED is not None:
        _rebind_wrapped_load_weights(_WRAPPED)
        return

    orig: Callable[..., Any] | None = None
    for mod in _iter_rebind_modules():
        candidate = getattr(mod, "load_weights", None)
        if candidate is None or not callable(candidate):
            continue
        if getattr(candidate, "_anthill_avatar_patched", None) is True:
            _WRAPPED = candidate
            _rebind_wrapped_load_weights(_WRAPPED)
            return
        if orig is None:
            orig = candidate

    if orig is None:
        return

    _WRAPPED = _make_wrapper(orig)
    _rebind_wrapped_load_weights(_WRAPPED)


def _iter_blockswap_rebind_modules():
    for name, mod in sys.modules.items():
        if mod is None:
            continue
        if any(_module_matches(name, suffix) for suffix in _BLOCKSWAP_REBIND_SUFFIXES):
            yield mod


def _rebind_wrapped_init_blockswap(wrapped: Callable[..., Any]) -> None:
    for mod in _iter_blockswap_rebind_modules():
        if hasattr(mod, "init_blockswap"):
            mod.init_blockswap = wrapped  # type: ignore[assignment]


def _blockswap_patched_linear(transformer: Any, block_swap_args: Any) -> None:
    if not getattr(transformer, "patched_linear", False) or block_swap_args is None:
        return
    transformer.block_swap(
        max(0, block_swap_args.get("blocks_to_swap", 0)),
        block_swap_args.get("offload_txt_emb", False),
        block_swap_args.get("offload_img_emb", False),
        vace_blocks_to_swap=block_swap_args.get("vace_blocks_to_swap", 0),
        prefetch_blocks=block_swap_args.get("prefetch_blocks", 0),
        block_swap_debug=block_swap_args.get("block_swap_debug", False),
    )


def _make_blockswap_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(orig)
    def init_blockswap(*args: Any, **kwargs: Any):
        if _should_skip_initial_blockswap():
            _state.deferred_blockswap_skipped = True
            _log_stage("defer block swap init", detail="until after VAE encode")
            return None
        result = orig(*args, **kwargs)
        if args:
            _blockswap_patched_linear(args[0], args[1] if len(args) > 1 else None)
        return result

    init_blockswap._anthill_avatar_patched = True  # type: ignore[attr-defined]
    return init_blockswap


def _patch_init_blockswap_defer() -> None:
    global _BLOCKSWAP_WRAPPED
    if _BLOCKSWAP_WRAPPED is not None:
        _rebind_wrapped_init_blockswap(_BLOCKSWAP_WRAPPED)
        return

    orig: Callable[..., Any] | None = None
    for mod in _iter_blockswap_rebind_modules():
        candidate = getattr(mod, "init_blockswap", None)
        if candidate is None or not callable(candidate):
            continue
        if getattr(candidate, "_anthill_avatar_patched", None) is True:
            _BLOCKSWAP_WRAPPED = candidate
            _rebind_wrapped_init_blockswap(_BLOCKSWAP_WRAPPED)
            return
        if orig is None:
            orig = candidate

    if orig is None:
        return

    _BLOCKSWAP_WRAPPED = _make_blockswap_wrapper(orig)
    _rebind_wrapped_init_blockswap(_BLOCKSWAP_WRAPPED)


def install_avatar_sampler_perf() -> None:
    """Patch WanVideoWrapper to keep GPU free during WanVAE encode."""
    _patch_load_weights_defer()
    _patch_init_blockswap_defer()
