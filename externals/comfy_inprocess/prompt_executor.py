"""Run Comfy API workflows via vendored ComfyUI PromptExecutor (Option C)."""

from __future__ import annotations

import os
import uuid
from typing import Any

from externals.comfy_inprocess.executor import (
    ComfyWorkflowError,
    find_node_id,
    strip_skipped_workflow_nodes,
)


class StubPromptServer:
    """Minimal server stand-in for headless PromptExecutor."""

    client_id = None
    last_node_id = None
    sockets_metadata: dict = {}

    def send_sync(self, *_args, **_kwargs) -> None:
        pass

    def send_progress_text(self, *_args, **_kwargs) -> None:
        pass


def should_use_comfy_executor(prompt: dict[str, Any] | None = None) -> bool:
    """Whether to run via vendored ComfyUI ``PromptExecutor``.

    Default (env unset): legacy topo executor for simple workflows (e.g. Qwen
    image2image, ~20s). PromptExecutor only for MEGA/VACE graphs that need it,
    unless ``AH_COMFY_EXECUTOR=comfy`` forces Comfy execution for all prompts.
    """
    raw = os.environ.get("AH_COMFY_EXECUTOR", "").strip().lower()
    if raw in ("legacy", "minimal", "simple", "0", "false", "no"):
        return False
    if raw in ("comfy", "prompt", "executor", "1", "true", "yes", "on"):
        return True
    if prompt is None:
        return False
    from externals.comfy_inprocess.comfy_memory import prompt_uses_mega_vace

    return prompt_uses_mega_vace(prompt)


def _prompt_executor_cache_args() -> dict[str, float]:
    """Cache RAM headroom keys required by vendored execution.PromptExecutor."""
    ram = float(os.environ.get("AH_COMFY_RAM_CACHE_GB", "16"))
    raw_inactive = os.environ.get("AH_COMFY_RAM_INACTIVE_CACHE_GB", "").strip()
    ram_inactive = float(raw_inactive) if raw_inactive else ram
    return {"ram": ram, "ram_inactive": ram_inactive}


def _output_node_ids(prompt: dict[str, Any]) -> list[str]:
    import nodes

    outputs: list[str] = []
    for nid, node in prompt.items():
        ctype = node.get("class_type")
        if not ctype:
            continue
        cls = nodes.NODE_CLASS_MAPPINGS.get(ctype)
        if cls is not None and getattr(cls, "OUTPUT_NODE", False):
            outputs.append(nid)
    decode = find_node_id(prompt, "VAEDecode")
    if decode and decode not in outputs:
        outputs.append(decode)
    if not outputs:
        raise ComfyWorkflowError("No output node (VAEDecode or OUTPUT_NODE) in workflow")
    return outputs


def _format_executor_failure(executor) -> str:
    """Build a readable message from PromptExecutor.status_messages."""
    parts: list[str] = []
    for event, data in getattr(executor, "status_messages", ()):
        if event != "execution_error":
            continue
        node = data.get("node_id", "?")
        ctype = data.get("node_type", "?")
        msg = (data.get("exception_message") or "").strip()
        etype = data.get("exception_type", "")
        tb = data.get("traceback") or []
        parts.append(f"node {node} ({ctype}): {etype}: {msg}")
        if tb:
            parts.append("".join(tb).rstrip())
    if parts:
        return "\n".join(parts)
    return "unknown error (no execution_error in status_messages)"


def _unwrap_output_slot(slot: Any) -> Any:
    """Unwrap PromptExecutor list batching (``merge_result_data`` → ``[tensor]`` per slot)."""
    while isinstance(slot, list):
        if len(slot) == 0:
            return None
        if len(slot) == 1:
            slot = slot[0]
            continue
        if all(hasattr(item, "detach") for item in slot):
            import torch

            return torch.cat(slot, dim=0)
        break
    return slot


def _cache_entry_to_outputs(entry: Any) -> tuple[Any, ...] | None:
    if entry is None:
        return None
    outputs = getattr(entry, "outputs", None)
    if outputs is None and isinstance(entry, (list, tuple)):
        outputs = entry
    if outputs is None:
        return None
    if isinstance(outputs, list):
        return tuple(_unwrap_output_slot(o) for o in outputs)
    return (_unwrap_output_slot(outputs),)


def _collect_tensor_outputs(executor, prompt: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Read node outputs from PromptExecutor cache after a successful run."""
    import asyncio

    cache = executor.caches.outputs
    out: dict[str, tuple[Any, ...]] = {}

    async def _collect_async() -> None:
        for nid in prompt:
            entry = None
            get_fn = getattr(cache, "get", None)
            if get_fn is not None:
                entry = await get_fn(nid)
            if entry is None:
                get_local = getattr(cache, "get_local", None)
                if get_local is not None:
                    entry = get_local(nid)
            converted = _cache_entry_to_outputs(entry)
            if converted is not None:
                out[nid] = converted

    asyncio.run(_collect_async())
    return out


def execute_prompt_comfy(
    prompt: dict[str, Any],
    *,
    nodes_module=None,
    output_node_ids: list[str] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Execute workflow with ComfyUI PromptExecutor; return node outputs map."""
    _ = nodes_module
    from externals.comfy_inprocess.comfy_memory import prompt_uses_mega_vace
    from externals.comfy_inprocess.node_registry import install_handler_nodes
    from externals.comfy_inprocess.wan_video_wrapper import wan_wrapper_enabled

    # ComfyI2VRunner.bootstrap_comfy() already set folder_paths input/output
    # (sessions/.../comfy_work/input). Re-bootstrapping with "." breaks LoadImage.
    load_wan = wan_wrapper_enabled() or (
        wan_wrapper_enabled(for_image2video=True) and prompt_uses_mega_vace(prompt)
    )
    install_handler_nodes(load_wan_wrapper=load_wan)

    from execution import CacheType, PromptExecutor

    server = StubPromptServer()
    cache_type = CacheType.CLASSIC
    raw_cache = os.environ.get("AH_COMFY_CACHE", "classic").strip().lower()
    if raw_cache in ("none", "0", "off"):
        # NullCache does not persist node outputs; Anthill must read VAEDecode after run.
        print(
            "$image2video: AH_COMFY_CACHE=none ignored for PromptExecutor "
            "(need classic cache to collect VAEDecode); using classic",
            flush=True,
        )
        cache_type = CacheType.CLASSIC
    elif raw_cache == "lru":
        cache_type = CacheType.LRU

    from externals.comfy_inprocess.comfy_memory import prompt_executor_memory_hooks
    from externals.comfy_inprocess.vram_config import apply_comfy_vram_settings

    apply_comfy_vram_settings()
    executor = PromptExecutor(
        server,
        cache_type=cache_type,
        cache_args=_prompt_executor_cache_args(),
    )
    run_prompt = strip_skipped_workflow_nodes(prompt)
    prompt_id = str(uuid.uuid4())
    targets = output_node_ids or _output_node_ids(run_prompt)
    with prompt_executor_memory_hooks(run_prompt):
        # Vendored execution.PromptExecutor.execute() returns None; use executor.success.
        executor.execute(run_prompt, prompt_id, execute_outputs=targets)
    if not executor.success:
        detail = _format_executor_failure(executor)
        raise ComfyWorkflowError(
            f"Comfy PromptExecutor failed (prompt_id={prompt_id}):\n{detail}"
        )
    outputs = _collect_tensor_outputs(executor, run_prompt)
    decode_id = find_node_id(run_prompt, "VAEDecode")
    if decode_id and decode_id not in outputs:
        raise ComfyWorkflowError(
            f"VAEDecode node {decode_id!r} produced no outputs "
            f"(prompt_id={prompt_id})"
        )
    return outputs
