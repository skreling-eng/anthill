"""Optional import stubs so comfy_lib loads without a full ComfyUI install.

Anthill in-process bootstrap loads only a small comfy_extras whitelist (see
``bootstrap._ANTHILL_COMFY_EXTRA_MODULES``). Newer ComfyUI extras need the
``comfy_api`` package and are skipped unless ``AH_COMFY_LOAD_ALL_EXTRAS=1``.
"""

from __future__ import annotations

import sys
import types


def ensure_comfy_import_stubs() -> None:
    """Inject minimal comfy_aimdo stubs when the package is not installed."""
    if "comfy_aimdo" in sys.modules:
        return
    if _try_import("comfy_aimdo"):
        return

    aimdo = types.ModuleType("comfy_aimdo")
    control = types.ModuleType("comfy_aimdo.control")
    host_buffer = types.ModuleType("comfy_aimdo.host_buffer")
    model_vbar = types.ModuleType("comfy_aimdo.model_vbar")
    model_mmap = types.ModuleType("comfy_aimdo.model_mmap")
    torch_mod = types.ModuleType("comfy_aimdo.torch")

    control.get_total_vram_usage = lambda: 0

    class HostBuffer:
        def __init__(self, size: int) -> None:
            self.size = size

    host_buffer.HostBuffer = HostBuffer

    def _noop(*_a, **_k):
        return None

    model_vbar.vbar_fault = _noop
    model_vbar.vbar_signature_compare = lambda *_a, **_k: True
    model_vbar.vbars_analyze = lambda: {}
    model_vbar.vbar_unpin = _noop
    model_vbar.ModelVBAR = object

    class ModelMMAP:
        def __init__(self, *_a, **_k) -> None:
            pass

    model_mmap.ModelMMAP = ModelMMAP
    torch_mod.aimdo_to_tensor = lambda v, device: v
    torch_mod.hostbuf_to_tensor = lambda buf: buf

    aimdo.control = control
    aimdo.host_buffer = host_buffer
    aimdo.model_vbar = model_vbar
    aimdo.model_mmap = model_mmap
    aimdo.torch = torch_mod

    sys.modules["comfy_aimdo"] = aimdo
    sys.modules["comfy_aimdo.control"] = control
    sys.modules["comfy_aimdo.host_buffer"] = host_buffer
    sys.modules["comfy_aimdo.model_vbar"] = model_vbar
    sys.modules["comfy_aimdo.model_mmap"] = model_mmap
    sys.modules["comfy_aimdo.torch"] = torch_mod


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
