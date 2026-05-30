"""Bootstrap comfy_lib for in-process Comfy workflow execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMFY_LIB = _REPO_ROOT / "comfy_lib"
_BOOTSTRAPPED = False


def comfy_lib_root() -> Path:
    return _COMFY_LIB.resolve()


def comfyui_models_root() -> Path | None:
    for key in ("AH_COMFYUI_ROOT", "COMFYUI_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            models = Path(raw).expanduser() / "models"
            if models.is_dir():
                return models.resolve()
    default = Path(r"G:\ComfyUI_V\models")
    if default.is_dir():
        return default.resolve()
    return None


def resolve_comfy_python() -> Path | None:
    for key in ("AH_COMFY_PYTHON", "COMFYUI_PYTHON", "COMFYUI_VENV_PYTHON"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_dir():
            for name in ("python.exe", "python"):
                candidate = path / "Scripts" / name if os.name == "nt" else path / "bin" / name
                if candidate.is_file():
                    return candidate.resolve()
        if path.is_file():
            return path.resolve()
    for candidate in (
        Path(r"G:\ComfyUI_V\.venv\Scripts\python.exe"),
        _REPO_ROOT / ".venvs" / "comfy" / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _add_checkpoint_roots() -> None:
    import folder_paths

    from externals.image.model_paths import models_roots

    seen: set[str] = set()
    for models_root in models_roots():
        for sub in ("", "checkpoints"):
            ckpt_dir = models_root / sub if sub else models_root
            if not ckpt_dir.is_dir():
                continue
            key = str(ckpt_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            folder_paths.add_model_folder_path("checkpoints", key)

        qwen = models_root / "qwen-rapid"
        if qwen.is_dir():
            key = str(qwen.resolve())
            if key not in seen:
                seen.add(key)
                folder_paths.add_model_folder_path("checkpoints", key)

    comfy_models = comfyui_models_root()
    if comfy_models is not None:
        for sub in ("checkpoints",):
            ckpt_dir = comfy_models / sub
            if ckpt_dir.is_dir():
                key = str(ckpt_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    folder_paths.add_model_folder_path("checkpoints", key)


def bootstrap_comfy(*, input_dir: Path, output_dir: Path) -> None:
    """Add comfy_lib to sys.path and configure folder_paths (once per process)."""
    global _BOOTSTRAPPED
    root = comfy_lib_root()
    if not root.is_dir():
        raise FileNotFoundError(f"comfy_lib not found at {root}")

    from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

    ensure_comfy_import_stubs()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import folder_paths

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(output_dir))
    _add_checkpoint_roots()

    if not _BOOTSTRAPPED:
        import nodes

        nodes.init_extra_nodes(init_custom_nodes=False)
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
