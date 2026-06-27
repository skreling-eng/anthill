"""Bootstrap comfy_lib for in-process Comfy workflow execution."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMFY_LIB = _REPO_ROOT / "comfy_lib"
_BOOTSTRAPPED = False

# comfy_extras modules that work without ComfyUI's separate ``comfy_api`` package.
# Full ``init_extra_nodes()`` pulls in dozens of newer nodes and spams import warnings.
_ANTHILL_COMFY_EXTRA_MODULES = (
    "nodes_model_advanced.py",  # ModelSamplingSD3 ($image2video workflows)
)

# Legacy Wan nodes registered via wan_legacy_nodes (comfy_extras/nodes_wan uses comfy_api v3).


def _load_all_comfy_extras() -> None:
    import nodes

    nodes.init_extra_nodes(init_custom_nodes=False)


def _load_minimal_comfy_extras() -> None:
    import nodes
    from nodes import load_custom_node

    extras_dir = comfy_lib_root() / "comfy_extras"
    failed: list[str] = []
    for name in _ANTHILL_COMFY_EXTRA_MODULES:
        path = extras_dir / name
        if not path.is_file():
            failed.append(name)
            continue
        if not load_custom_node(str(path), module_parent="comfy_extras"):
            failed.append(name)
    if failed:
        logging.warning(
            "Anthill comfy_extras not loaded (optional for some workflows): %s",
            ", ".join(failed),
        )


def _init_comfy_extras() -> None:
    raw = os.environ.get("AH_COMFY_LOAD_ALL_EXTRAS", "").strip().lower()
    if raw in ("1", "true", "yes", "on", "all"):
        _load_all_comfy_extras()
    else:
        _load_minimal_comfy_extras()


def comfy_lib_root() -> Path:
    return _COMFY_LIB.resolve()


def _register_anthill_model_paths() -> None:
    """Point comfy_lib folder_paths at repo models/ (and optional MODELS_PATH roots)."""
    import folder_paths

    from externals.image.model_paths import models_roots

    roots = models_roots()
    if not roots:
        return

    folder_paths.models_dir = str(roots[0].resolve())
    seen: set[tuple[str, str]] = set()

    def _add(folder_key: str, path: Path) -> None:
        if not path.is_dir():
            return
        key = (folder_key, str(path.resolve()))
        if key in seen:
            return
        seen.add(key)
        folder_paths.add_model_folder_path(folder_key, str(path.resolve()), is_default=True)

    for root in roots:
        for sub in ("", "checkpoints"):
            _add("checkpoints", root / sub if sub else root)
        _add("checkpoints", root / "qwen-rapid")
        _add("checkpoints", root / "wan")
        _add("diffusion_models", root / "diffusion_models")
        _add("diffusion_models", root / "flux2klein")
        _add("text_encoders", root / "text_encoders")
        _add("vae", root / "vae")
        _add("clip_vision", root / "clip_vision")
        _add("controlnet", root / "controlnet")
        _add("wav2vec2", root / "wav2vec2")

        qwen = root / "qwen-image"
        if qwen.is_dir():
            for sub in ("diffusion_models", "text_encoders", "vae", "controlnet"):
                _add(sub, qwen / sub)


def _ensure_headless_prompt_server() -> None:
    """Patch ComfyUI server stub before WanVideoWrapper imports ``PromptServer``."""
    try:
        import server
    except ImportError:
        return

    cls = server.PromptServer
    if not hasattr(cls, "send_progress_text"):

        def send_progress_text(self, *_args, **_kwargs) -> None:
            pass

        cls.send_progress_text = send_progress_text  # type: ignore[method-assign]

    inst = getattr(cls, "instance", None)
    if inst is not None and not hasattr(inst, "send_progress_text"):
        inst.send_progress_text = lambda *_a, **_k: None  # type: ignore[method-assign]


def bootstrap_comfy(
    *,
    input_dir: Path,
    output_dir: Path,
    load_wan_wrapper: bool = False,
    vram_profile: str = "default",
) -> None:
    """Add comfy_lib to sys.path and configure folder_paths (once per process)."""
    global _BOOTSTRAPPED
    root = comfy_lib_root()
    if not root.is_dir():
        raise FileNotFoundError(f"comfy_lib not found at {root}")

    from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

    ensure_comfy_import_stubs()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if vram_profile == "image2image":
        from externals.comfy_inprocess.vram_config import apply_image2image_vram_settings

        apply_image2image_vram_settings()
    elif vram_profile == "avatar":
        from externals.comfy_inprocess.vram_config import apply_avatar_vram_settings

        apply_avatar_vram_settings()
    else:
        from externals.comfy_inprocess.vram_config import apply_comfy_vram_settings

        apply_comfy_vram_settings()

    _ensure_headless_prompt_server()

    import folder_paths

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(output_dir))
    _register_anthill_model_paths()

    if load_wan_wrapper:
        from externals.comfy_inprocess.audio_nodes import register_comfy_audio_handlers

        register_comfy_audio_handlers()

    if not _BOOTSTRAPPED:
        _init_comfy_extras()
        from externals.comfy_inprocess.wan_video_wrapper import (
            load_wan_video_wrapper,
            wan_wrapper_enabled,
        )

        if load_wan_wrapper and wan_wrapper_enabled(for_image2video=True):
            load_wan_video_wrapper()
        from externals.comfy_inprocess.wan_legacy_nodes import register_wan_legacy_nodes

        register_wan_legacy_nodes()
        _BOOTSTRAPPED = True


def get_nodes_module():
    if not _BOOTSTRAPPED:
        bootstrap_comfy(
            input_dir=Path(
                os.environ.get("AH_COMFY_INPUT_DIR", _REPO_ROOT / "comfy_lib" / "input")
            ),
            output_dir=Path(
                os.environ.get("AH_COMFY_OUTPUT_DIR", _REPO_ROOT / "comfy_lib" / "output")
            ),
        )
    import nodes

    return nodes
