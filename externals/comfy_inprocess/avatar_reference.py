"""Inject anchor frames as SkyReels reference_video between $avatar passes."""

from __future__ import annotations

from typing import Any

_REFERENCE_FRAMES: Any | None = None
_REGISTERED = False


def set_reference_video_frames(frames: Any) -> None:
    """Store decoded anchor frames as Comfy IMAGE batch (T, H, W, C) in [0, 1]."""
    global _REFERENCE_FRAMES
    _REFERENCE_FRAMES = frames


def clear_reference_video_frames() -> None:
    global _REFERENCE_FRAMES
    _REFERENCE_FRAMES = None


def _reference_video_handler(_inputs: dict[str, Any]) -> tuple[Any, ...]:
    if _REFERENCE_FRAMES is None:
        raise RuntimeError(
            "AnthillAvatarReferenceVideo: no anchor frames set for pass 2"
        )
    return (_REFERENCE_FRAMES,)


def register_avatar_reference_handler() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from externals.comfy_inprocess.executor import register_node_handler

    register_node_handler(
        "AnthillAvatarReferenceVideo",
        _reference_video_handler,
        input_types={"required": {}, "optional": {}},
    )
    _REGISTERED = True
