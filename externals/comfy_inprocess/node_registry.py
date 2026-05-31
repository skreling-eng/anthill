"""Register Anthill node handlers into nodes.NODE_CLASS_MAPPINGS for PromptExecutor."""

from __future__ import annotations

from typing import Any, Callable


def _make_handler_node(class_type: str, handler: Callable[[dict[str, Any]], tuple]) -> type:
    """Wrap a handler(inputs) -> tuple as a legacy Comfy node class."""

    class HandlerNode:
        FUNCTION = "execute"
        OUTPUT_NODE = False

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}, "optional": {}}

        def execute(self, **kwargs):
            return handler(kwargs)

    HandlerNode.__name__ = class_type
    return HandlerNode


def install_handler_nodes() -> None:
    """Merge Anthill-only handlers into NODE_CLASS_MAPPINGS (idempotent)."""
    import nodes
    from externals.comfy_inprocess.executor import _NODE_HANDLERS
    from externals.comfy_inprocess.wan_legacy_nodes import register_wan_legacy_nodes
    from externals.comfy_inprocess.wan_video_wrapper import (
        load_wan_video_wrapper,
        wan_wrapper_enabled,
    )
    from externals.image2video.comfy_nodes import register_i2v_node_handlers

    if wan_wrapper_enabled(for_image2video=True):
        load_wan_video_wrapper()
    register_wan_legacy_nodes()
    register_i2v_node_handlers()
    for class_type, handler in _NODE_HANDLERS.items():
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            nodes.NODE_CLASS_MAPPINGS[class_type] = _make_handler_node(class_type, handler)
