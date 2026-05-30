"""Execute Comfy API-format workflow JSON using comfy_lib nodes."""

from __future__ import annotations

from collections import deque
from typing import Any

from externals.image2image.comfy_qwen_nodes import text_encode_qwen_image_edit_plus


class ComfyWorkflowError(RuntimeError):
    pass


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
    if class_type == "TextEncodeQwenImageEditPlus":
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


def _normalize_outputs(result: Any) -> tuple[Any, ...]:
    if result is None:
        return (None,)
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    if isinstance(result, tuple):
        return result
    return (result,)


_SKIP_NODES = frozenset({"SaveImage", "PreviewImage"})


def execute_prompt(
    prompt: dict[str, Any],
    *,
    nodes_module,
    output_node_ids: list[str] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Run a Comfy /prompt-style workflow dict; return all node outputs."""
    if output_node_ids is None:
        output_node_ids = []
    order = _topo_order(prompt)
    outputs: dict[str, tuple[Any, ...]] = {}
    mappings = nodes_module.NODE_CLASS_MAPPINGS

    for nid in order:
        node = prompt[nid]
        class_type = node.get("class_type")
        if not class_type:
            raise ComfyWorkflowError(f"Node {nid} missing class_type")
        inputs = _resolve_inputs(node.get("inputs") or {}, outputs)

        if class_type in _SKIP_NODES:
            outputs[nid] = (None,)
            continue
        elif class_type == "TextEncodeQwenImageEditPlus":
            result = (
                text_encode_qwen_image_edit_plus(
                    inputs["clip"],
                    inputs.get("prompt", ""),
                    inputs.get("vae"),
                    image1=inputs.get("image1"),
                    image2=inputs.get("image2"),
                    image3=inputs.get("image3"),
                ),
            )
        elif class_type not in mappings:
            # Orphan utility nodes (e.g. unused PrimitiveInt) — skip quietly.
            outputs[nid] = (None,)
            continue
        else:
            cls = mappings[class_type]
            instance = cls()
            func = getattr(instance, cls.FUNCTION)
            filtered = _filter_node_inputs(class_type, cls, inputs)
            result = func(**filtered)

        outputs[nid] = _normalize_outputs(result)

    return outputs


def find_node_id(prompt: dict[str, Any], class_type: str) -> str | None:
    for nid, node in prompt.items():
        if node.get("class_type") == class_type:
            return nid
    return None
