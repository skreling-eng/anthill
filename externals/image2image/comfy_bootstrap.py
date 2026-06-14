"""Bootstrap comfy_lib for in-process Qwen-Rapid-AIO inference."""

from externals.comfy_inprocess.bootstrap import (  # noqa: F401
    bootstrap_comfy,
    comfy_lib_root,
    comfyui_models_root,
    get_nodes_module,
)
