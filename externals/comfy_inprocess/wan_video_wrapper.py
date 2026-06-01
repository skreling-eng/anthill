"""Load vendored ComfyUI-WanVideoWrapper into comfy_lib NODE_CLASS_MAPPINGS."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOADED = False
_LOAD_FAILED = False


def wan_wrapper_path() -> Path:
    from externals.comfy_inprocess.bootstrap import comfy_lib_root

    return comfy_lib_root() / "ComfyUI-WanVideoWrapper"


def wan_wrapper_enabled(*, for_image2video: bool = False) -> bool:
    """Whether to import ComfyUI-WanVideoWrapper (MEGA VACE nodes, etc.)."""
    raw = os.environ.get("AH_COMFY_WAN_WRAPPER", "").strip().lower()
    if raw in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    return for_image2video


def wan_wrapper_loaded() -> bool:
    return _LOADED


def load_wan_video_wrapper(*, force: bool = False) -> bool:
    """Register WanVideoWrapper nodes via comfy_lib ``load_custom_node``."""
    global _LOADED, _LOAD_FAILED
    if _LOADED:
        return True
    if _LOAD_FAILED and not force:
        return False

    path = wan_wrapper_path()
    if not path.is_dir():
        logging.warning("ComfyUI-WanVideoWrapper not found at %s", path)
        _LOAD_FAILED = True
        return False

    if not _try_import("gguf"):
        _LOAD_FAILED = True
        import sys

        logging.warning(
            "ComfyUI-WanVideoWrapper skipped (missing gguf; python=%s). "
            "Install deps in .venvs/comfy-wan: "
            "UV_PROJECT_ENVIRONMENT=.venvs/comfy-wan uv sync --extra media,comfy-wan,clip "
            "(needs gguf). Set AH_EXTERNAL_VENV_image2video=.venvs/comfy-wan and restart worker.",
            sys.executable,
        )
        return False

    import nodes
    from nodes import load_custom_node

    ok = load_custom_node(str(path), module_parent="custom_nodes")
    if ok:
        _LOADED = True
        logging.info(
            "Loaded ComfyUI-WanVideoWrapper (%d nodes)",
            sum(
                1
                for name in nodes.NODE_CLASS_MAPPINGS
                if name.startswith("WanVideo") or "WanVideo" in name
            ),
        )
    else:
        _LOAD_FAILED = True
        logging.warning("Failed to load ComfyUI-WanVideoWrapper from %s", path)
    return ok


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
