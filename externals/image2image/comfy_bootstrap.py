"""Bootstrap comfy_lib for in-process Qwen-Rapid-AIO inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMFY_LIB = _REPO_ROOT / "comfy_lib"
_BOOTSTRAPPED = False


def comfy_lib_root() -> Path:
    return _COMFY_LIB.resolve()


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


def bootstrap_comfy(*, input_dir: Path, output_dir: Path) -> None:
    """Add comfy_lib to sys.path and configure folder_paths (once per process)."""
    global _BOOTSTRAPPED
    root = comfy_lib_root()
    if not root.is_dir():
        raise FileNotFoundError(f"comfy_lib not found at {root}")

    from externals.image2image.comfy_stubs import ensure_comfy_import_stubs

    ensure_comfy_import_stubs()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import folder_paths
    from externals.image.model_paths import models_roots

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_paths.set_input_directory(str(input_dir))
    folder_paths.set_output_directory(str(output_dir))

    for models_root in models_roots():
        ckpt_dir = models_root / "qwen-rapid"
        if ckpt_dir.is_dir():
            folder_paths.add_model_folder_path("checkpoints", str(ckpt_dir.resolve()))

    if not _BOOTSTRAPPED:
        import nodes  # noqa: F401 — registers NODE_CLASS_MAPPINGS

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
