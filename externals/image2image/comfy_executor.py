"""Execute Comfy API-format workflow JSON using comfy_lib nodes."""

from __future__ import annotations

from externals.comfy_inprocess.executor import (  # noqa: F401
    ComfyWorkflowError,
    execute_prompt,
    find_node_id,
    register_node_handler,
    topo_order as _topo_order,
)
from externals.image2image.comfy_qwen_nodes import text_encode_qwen_image_edit_plus


def _qwen_encode_handler(inputs: dict) -> tuple:
    return (
        text_encode_qwen_image_edit_plus(
            inputs["clip"],
            inputs.get("prompt", ""),
            inputs.get("vae"),
            image1=inputs.get("image1"),
            image2=inputs.get("image2"),
            image3=inputs.get("image3"),
        ),
    )


register_node_handler("TextEncodeQwenImageEditPlus", _qwen_encode_handler)
