"""Register Anthill node handlers into nodes.NODE_CLASS_MAPPINGS for PromptExecutor."""

from __future__ import annotations

from typing import Any, Callable


def _make_handler_node(
    class_type: str,
    handler: Callable[[dict[str, Any]], tuple],
    *,
    input_types: dict[str, Any],
) -> type:
    """Wrap a handler(inputs) -> tuple as a legacy Comfy node class."""

    class HandlerNode:
        FUNCTION = "execute"
        OUTPUT_NODE = False

        @classmethod
        def INPUT_TYPES(cls):
            return input_types

        def execute(self, **kwargs):
            return handler(kwargs)

    HandlerNode.__name__ = class_type
    return HandlerNode


def install_handler_nodes(*, load_wan_wrapper: bool = False) -> None:
    """Merge Anthill-only handlers into NODE_CLASS_MAPPINGS (idempotent)."""
    import nodes
    from externals.comfy_inprocess.executor import _NODE_HANDLERS, handler_input_types
    from externals.comfy_inprocess.wan_legacy_nodes import register_wan_legacy_nodes
    from externals.comfy_inprocess.wan_video_wrapper import load_wan_video_wrapper
    from externals.image2video.comfy_nodes import register_i2v_node_handlers

    if load_wan_wrapper:
        load_wan_video_wrapper()
    register_wan_legacy_nodes()
    register_i2v_node_handlers()
    for class_type, handler in _NODE_HANDLERS.items():
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            nodes.NODE_CLASS_MAPPINGS[class_type] = _make_handler_node(
                class_type,
                handler,
                input_types=handler_input_types(class_type),
            )
