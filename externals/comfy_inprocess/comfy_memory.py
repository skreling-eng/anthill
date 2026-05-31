"""ComfyUI-style VRAM lifecycle for in-process workflow execution."""

from __future__ import annotations

import contextvars
import gc
from typing import Any

_PROMPT_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "ah_comfy_prompt", default=None
)
_LAST_NODE: dict[str, str | None] = {"class_type": None}

# Nodes that should not leave the previous GPU resident set on the next heavy step.
_UNLOAD_BEFORE: frozenset[str] = frozenset(
    {
        "KSampler",
        "VAEDecode",
        "VAEEncode",
        "VAEDecodeTiled",
        "VAEEncodeTiled",
    }
)

# Nodes that tend to finish with CLIP/VAE/UNet still loaded (MEGA I2V path).
_RELEASE_AFTER: frozenset[str] = frozenset(
    {
        "CLIPTextEncode",
        "CLIPVisionEncode",
        "WanImageToVideo",
        "WanVaceToVideo",
        "WanVideoVACEStartToEndFrame",
        "KSampler",
    }
)


def prompt_uses_mega_vace(prompt: dict[str, Any]) -> bool:
    return any(n.get("class_type") == "WanVaceToVideo" for n in prompt.values())


def comfy_memory_enabled(prompt: dict[str, Any]) -> bool:
    import os

    raw = os.environ.get("AH_COMFY_MEMORY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on", "comfy", "mega"):
        return True
    # Default: Comfy-like unload for MEGA/VACE workflows only.
    return prompt_uses_mega_vace(prompt)


def _release_gpu(*, force: bool = False) -> None:
    gc.collect()
    try:
        import comfy.model_management as mm

        if force:
            mm.unload_all_models()
        mm.soft_empty_cache(force=True)
        if force:
            mm.cleanup_models_gc()
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare_node(class_type: str, *, enabled: bool) -> None:
    if not enabled or class_type not in _UNLOAD_BEFORE:
        return
    _release_gpu(force=True)


def finalize_node(class_type: str, *, enabled: bool) -> None:
    if not enabled:
        return
    if class_type in _RELEASE_AFTER:
        _release_gpu(force=True)
    elif class_type in _UNLOAD_BEFORE:
        _release_gpu(force=False)


def load_vae_for_encode(vae: Any, *, length: int, height: int, width: int) -> None:
    """Mirror Comfy VAE.encode: load patcher to GPU with an encode memory estimate."""
    import comfy.model_management as mm

    pixel_shape = (1, length, height, width, 3)
    memory_used = vae.memory_used_encode(pixel_shape, vae.vae_dtype)
    mm.load_models_gpu(
        [vae.patcher],
        memory_required=memory_used,
        force_full_load=getattr(vae, "disable_offload", False),
    )


def handle_execution_oom() -> None:
    """Same recovery ComfyUI uses on CUDA OOM during a node."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except ImportError:
        pass
    _release_gpu(force=False)


def prompt_executor_memory_hooks(prompt: dict[str, Any]):
    """Context manager: unload GPU between heavy nodes when using PromptExecutor."""
    from contextlib import contextmanager

    import nodes
    from comfy_execution.utils import get_executing_context

    @contextmanager
    def _cm():
        enabled = comfy_memory_enabled(prompt)
        token = _PROMPT_CTX.set(prompt)
        _LAST_NODE["class_type"] = None
        original = nodes.before_node_execution

        def before_node_execution() -> None:
            if enabled and _LAST_NODE["class_type"]:
                finalize_node(_LAST_NODE["class_type"], enabled=True)
            original()
            if not enabled:
                return
            ctx = get_executing_context()
            if ctx is None:
                return
            node = prompt.get(ctx.node_id) or {}
            class_type = node.get("class_type") or ""
            _LAST_NODE["class_type"] = class_type
            prepare_node(class_type, enabled=True)

        nodes.before_node_execution = before_node_execution
        try:
            yield
        finally:
            nodes.before_node_execution = original
            if enabled and _LAST_NODE["class_type"]:
                finalize_node(_LAST_NODE["class_type"], enabled=True)
                _LAST_NODE["class_type"] = None
            _PROMPT_CTX.reset(token)
            if enabled:
                _release_gpu(force=True)

    return _cm()
