"""Execute Comfy API-format workflow JSON using comfy_lib nodes."""

from __future__ import annotations

import copy
import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

_NODE_HANDLERS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {}
# PromptExecutor only passes inputs listed in INPUT_TYPES (links still resolve).
_HANDLER_INPUT_TYPES: dict[str, dict[str, Any]] = {}
SKIP_NODES = frozenset({"SaveImage", "PreviewImage", "VHS_VideoCombine"})
_SKIP_NODES = SKIP_NODES  # alias for internal use


def strip_skipped_workflow_nodes(prompt: dict[str, Any]) -> dict[str, Any]:
    """Drop nodes Anthill handles elsewhere (VHS export, previews).

    PromptExecutor still walks every node id in the prompt; unregistered types
    like ``VHS_VideoCombine`` cause KeyError in ``execution._is_intermediate_output``.
    """
    removed = [
        nid
        for nid, node in prompt.items()
        if (node.get("class_type") or "") in SKIP_NODES
    ]
    if not removed:
        return prompt
    out = copy.deepcopy(prompt)
    for nid in removed:
        del out[nid]
    return out


class ComfyWorkflowError(RuntimeError):
    pass


def register_node_handler(
    class_type: str,
    handler: Callable[[dict[str, Any]], tuple[Any, ...]],
    *,
    input_types: dict[str, Any] | None = None,
) -> None:
    _NODE_HANDLERS[class_type] = handler
    if input_types is not None:
        _HANDLER_INPUT_TYPES[class_type] = input_types


def handler_input_types(class_type: str) -> dict[str, Any]:
    return _HANDLER_INPUT_TYPES.get(class_type, {"required": {}, "optional": {}})


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[1], int)
        and (isinstance(value[0], str) or isinstance(value[0], int))
    )


def _topo_order(prompt: dict[str, Any]) -> list[str]:
    deps: dict[str, set[str]] = {nid: set() for nid in prompt}
    for nid, node in prompt.items():
        inputs = node.get("inputs") or {}
        for val in inputs.values():
            if _is_link(val):
                deps[nid].add(str(val[0]))
    order: list[str] = []
    ready = deque(sorted(nid for nid, d in deps.items() if not d))
    while ready:
        nid = ready.popleft()
        order.append(nid)
        for other, d in deps.items():
            if nid in d:
                d.remove(nid)
                if not d:
                    ready.append(other)
    if len(order) != len(prompt):
        raise ComfyWorkflowError("Workflow graph has a cycle")
    return order


def _resolve_inputs(
    node_inputs: dict[str, Any],
    outputs: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, val in node_inputs.items():
        if key.startswith("_"):
            continue
        if _is_link(val):
            src_id, slot = str(val[0]), val[1]
            if src_id not in outputs:
                raise ComfyWorkflowError(f"Missing output from node {src_id!r}")
            out = outputs[src_id]
            if slot >= len(out):
                raise ComfyWorkflowError(
                    f"Node {src_id!r} output slot {slot} out of range ({len(out)})"
                )
            resolved[key] = out[slot]
        else:
            resolved[key] = val
    return resolved


def _filter_node_inputs(class_type: str, cls, inputs: dict[str, Any]) -> dict[str, Any]:
    if class_type in _NODE_HANDLERS:
        return inputs
    if not hasattr(cls, "INPUT_TYPES"):
        return inputs
    spec = cls.INPUT_TYPES()
    allowed: set[str] = set()
    for section in ("required", "optional", "hidden"):
        block = spec.get(section)
        if isinstance(block, dict):
            allowed.update(block.keys())
    if not allowed:
        return {k: v for k, v in inputs.items() if not k.startswith("+")}
    return {k: v for k, v in inputs.items() if k in allowed}


def _inject_hidden_inputs(
    nid: str,
    cls,
    inputs: dict[str, Any],
    *,
    prompt: dict[str, Any],
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject Comfy hidden inputs (unique_id, prompt, …) like execution.get_input_data."""
    if not hasattr(cls, "INPUT_TYPES"):
        return inputs
    hidden = cls.INPUT_TYPES().get("hidden")
    if not isinstance(hidden, dict):
        return inputs
    extra = extra_data or {}
    out = dict(inputs)
    for key, token in hidden.items():
        if key in out:
            continue
        if token == "PROMPT":
            out[key] = prompt
        elif token == "DYNPROMPT":
            out[key] = None
        elif token == "EXTRA_PNGINFO":
            out[key] = extra.get("extra_pnginfo")
        elif token == "UNIQUE_ID":
            out[key] = nid
        elif token == "AUTH_TOKEN_COMFY_ORG":
            out[key] = extra.get("auth_token_comfy_org")
        elif token == "API_KEY_COMFY_ORG":
            out[key] = extra.get("api_key_comfy_org")
    return out


def _normalize_outputs(result: Any) -> tuple[Any, ...]:
    if result is None:
        return (None,)
    if hasattr(result, "args"):
        args = getattr(result, "args")
        if isinstance(args, tuple):
            return args
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    if isinstance(result, tuple):
        return result
    return (result,)


def execute_prompt(
    prompt: dict[str, Any],
    *,
    nodes_module,
    output_node_ids: list[str] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Run a Comfy /prompt-style workflow dict; return all node outputs."""
    if should_use_comfy_executor(prompt):
        try:
            from externals.comfy_inprocess.prompt_executor import execute_prompt_comfy

            return execute_prompt_comfy(
                prompt,
                nodes_module=nodes_module,
                output_node_ids=output_node_ids,
            )
        except ImportError as exc:
            import logging

            logging.warning(
                "AH_COMFY_EXECUTOR=comfy unavailable (%s); using legacy executor",
                exc,
            )

    return _execute_prompt_legacy(
        prompt,
        nodes_module=nodes_module,
        output_node_ids=output_node_ids,
    )


def _execute_prompt_legacy(
    prompt: dict[str, Any],
    *,
    nodes_module,
    output_node_ids: list[str] | None = None,
    stop_before_class: str | None = None,
    only_classes: frozenset[str] | None = None,
    initial_outputs: dict[str, tuple[Any, ...]] | None = None,
    prepare_ksampler: bool = True,
) -> dict[str, tuple[Any, ...]]:
    """Legacy topo executor (fallback when PromptExecutor is disabled or fails)."""
    _ = output_node_ids
    from externals.comfy_inprocess.comfy_memory import (
        comfy_memory_enabled,
        finalize_node,
        handle_execution_oom,
        prepare_node,
        prompt_uses_flux2_klein,
        prompt_uses_qwen_image_edit,
    )

    order = _topo_order(prompt)
    outputs: dict[str, tuple[Any, ...]] = dict(initial_outputs or {})
    mappings = nodes_module.NODE_CLASS_MAPPINGS
    memory_mode = comfy_memory_enabled(prompt)
    qwen_edit = prompt_uses_qwen_image_edit(prompt)
    flux2_klein = prompt_uses_flux2_klein(prompt)
    klein_edit = flux2_klein and any(
        n.get("class_type") == "ReferenceLatent" for n in prompt.values()
    )
    klein_label = "$flux2_klein" if flux2_klein else "$image2image"
    timing = os.environ.get("AH_IMAGE2IMAGE_TIMING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    flux2_encode_ready = False

    for nid in order:
        node = prompt[nid]
        class_type = node.get("class_type")
        if not class_type:
            raise ComfyWorkflowError(f"Node {nid} missing class_type")
        if stop_before_class and class_type == stop_before_class:
            break
        if only_classes is not None and class_type not in only_classes:
            continue
        inputs = _resolve_inputs(node.get("inputs") or {}, outputs)

        if class_type in _SKIP_NODES:
            outputs[nid] = (None,)
            continue

        prepare_node(class_type, enabled=memory_mode)
        ksampler_heavy = class_type == "KSampler" and (qwen_edit or flux2_klein)
        t_node = time.perf_counter() if (timing or ksampler_heavy) else 0.0
        if (
            flux2_klein
            and not flux2_encode_ready
            and class_type in ("VAEEncode", "CLIPTextEncode")
        ):
            try:
                from externals.flux2_klein.comfy_sample_prep import prepare_for_klein_encode

                prepare_for_klein_encode(
                    clip=inputs.get("clip"),
                    vae=inputs.get("vae"),
                )
                flux2_encode_ready = True
            except ImportError:
                pass
        if ksampler_heavy and prepare_ksampler:
            steps = (node.get("inputs") or {}).get("steps", "?")
            sampler = (node.get("inputs") or {}).get("sampler_name", "?")
            print(
                f"{klein_label}: KSampler ({steps} steps, {sampler})…",
                flush=True,
            )
            if flux2_klein:
                try:
                    from externals.flux2_klein.comfy_sample_prep import (
                        offload_klein_conditioning,
                        offload_klein_image_tensor,
                        offload_klein_latent,
                    )

                    keep_refs = klein_edit
                    for key in ("positive", "negative"):
                        if key in inputs:
                            inputs[key] = offload_klein_conditioning(
                                inputs[key],
                                keep_reference_latents=keep_refs,
                            )
                    for src_id, out in list(outputs.items()):
                        src = prompt.get(src_id) or {}
                        ctype = src.get("class_type")
                        if ctype in ("CLIPTextEncode", "ReferenceLatent") and out:
                            outputs[src_id] = (
                                offload_klein_conditioning(
                                    out[0],
                                    keep_reference_latents=keep_refs,
                                ),
                                *out[1:],
                            )
                        elif ctype == "LoadImage" and out:
                            outputs[src_id] = (offload_klein_image_tensor(out[0]), *out[1:])
                        elif ctype == "VAEEncode" and out:
                            outputs[src_id] = (offload_klein_latent(out[0]), *out[1:])
                except ImportError:
                    pass
            model = inputs.get("model")
            if model is not None:
                try:
                    if flux2_klein:
                        from externals.flux2_klein.comfy_sample_prep import (
                            prepare_for_ksampler as _klein_prep,
                        )

                        _klein_prep(
                            model,
                            label=klein_label,
                            prefer_full_unet=klein_edit,
                        )
                    else:
                        from externals.image2image.comfy_sample_prep import (
                            prepare_for_ksampler,
                        )

                        prepare_for_ksampler(model)
                except ImportError:
                    pass

        try:
            if class_type in _NODE_HANDLERS:
                outputs[nid] = _NODE_HANDLERS[class_type](inputs)
            elif class_type in mappings:
                cls = mappings[class_type]
                instance = cls()
                func = getattr(instance, cls.FUNCTION)
                filtered = _filter_node_inputs(class_type, cls, inputs)
                filtered = _inject_hidden_inputs(
                    nid, cls, filtered, prompt=prompt
                )
                result = func(**filtered)
                outputs[nid] = _normalize_outputs(result)
            else:
                outputs[nid] = (None,)
        except Exception as exc:
            err = str(exc).lower()
            if "outofmemoryerror" in err or "out of memory" in err:
                handle_execution_oom()
            raise
        finally:
            if flux2_klein and class_type == "UNETLoader" and nid in outputs:
                try:
                    from externals.flux2_klein.comfy_sample_prep import park_unet_off_gpu

                    park_unet_off_gpu(outputs[nid][0])
                except ImportError:
                    pass
            finalize_node(class_type, enabled=memory_mode)
            if ksampler_heavy:
                elapsed = time.perf_counter() - t_node
                print(
                    f"{klein_label}: KSampler finished ({elapsed:.1f}s)",
                    flush=True,
                )
            elif timing and elapsed >= 0.5:
                print(
                    f"$image2image: node {class_type} ({nid}) {elapsed:.1f}s",
                    flush=True,
                )

    return outputs


def should_use_comfy_executor(prompt: dict[str, Any] | None = None) -> bool:
    from externals.comfy_inprocess.prompt_executor import should_use_comfy_executor as _should

    return _should(prompt)


# Keep public name for callers that imported execute_prompt body.
def execute_prompt_legacy(
    prompt: dict[str, Any],
    *,
    nodes_module,
    output_node_ids: list[str] | None = None,
    stop_before_class: str | None = None,
    only_classes: frozenset[str] | None = None,
    initial_outputs: dict[str, tuple[Any, ...]] | None = None,
    prepare_ksampler: bool = True,
) -> dict[str, tuple[Any, ...]]:
    return _execute_prompt_legacy(
        prompt,
        nodes_module=nodes_module,
        output_node_ids=output_node_ids,
        stop_before_class=stop_before_class,
        only_classes=only_classes,
        initial_outputs=initial_outputs,
        prepare_ksampler=prepare_ksampler,
    )


def find_node_id(prompt: dict[str, Any], class_type: str) -> str | None:
    for nid, node in prompt.items():
        if node.get("class_type") == class_type:
            return nid
    return None


# Back-compat alias for tests and callers.
topo_order = _topo_order
