"""Isolated venv (.venvs/custom_actions) for &action codegen and execution."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_NAME = "anthill"
_VENV_REL = Path(".venvs/custom_actions")
_SYNC_MARKER = ".anthill_synced"
_MODEL_MARKER = ".code_model_ready"

# Top-level module name -> PyPI package (when different).
_MODULE_TO_PYPI: dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "Image": "Pillow",
    "ImageDraw": "Pillow",
    "ImageFont": "Pillow",
}

_SKIP_MODULES = frozenset(
    {
        "ahlib",
        "externals",
        "anthill",
        _REPO_NAME,
    }
)


def _stdlib_modules() -> frozenset[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return frozenset(names)
    return frozenset(
        {
            "json",
            "math",
            "re",
            "pathlib",
            "shutil",
            "uuid",
            "os",
            "sys",
            "typing",
            "collections",
            "itertools",
            "functools",
            "copy",
            "hashlib",
            "tempfile",
            "io",
            "abc",
            "dataclasses",
            "enum",
            "struct",
            "base64",
            "datetime",
            "time",
            "random",
            "statistics",
            "decimal",
            "fractions",
            "contextlib",
            "warnings",
            "traceback",
            "subprocess",
        }
    )


def custom_venv_dir(repo_root: Path) -> Path:
    raw = os.environ.get("AH_CUSTOM_ACTIONS_VENV", "").strip()
    rel = Path(raw) if raw else _VENV_REL
    if rel.is_absolute():
        return rel.resolve()
    return (repo_root / rel).resolve()


def _venv_python(venv: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = venv / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No python in custom-actions venv: {venv}")


def venv_python(repo_root: Path) -> Path:
    return _venv_python(custom_venv_dir(repo_root))


def _uv_bin() -> str:
    import shutil

    return shutil.which(os.environ.get("UV", "uv")) or "uv"


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    root = str(repo_root.resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def ensure_venv(repo_root: Path, *, force_sync: bool = False) -> Path:
    """Create .venvs/custom_actions and uv sync --extra custom_actions."""
    venv = custom_venv_dir(repo_root)
    py = _venv_python(venv) if (venv / "pyvenv.cfg").is_file() else None
    marker = venv / _SYNC_MARKER
    if (
        not force_sync
        and os.environ.get("AH_FORCE_CUSTOM_VENV_SYNC", "").strip().lower()
        not in ("1", "true", "yes")
        and py is not None
        and py.is_file()
        and marker.is_file()
    ):
        return py

    venv.parent.mkdir(parents=True, exist_ok=True)
    if not (venv / "pyvenv.cfg").is_file():
        subprocess.run(
            [_uv_bin(), "venv", str(venv)],
            cwd=repo_root,
            check=True,
        )

    env = os.environ.copy()
    rel_venv = os.environ.get("AH_CUSTOM_ACTIONS_VENV", "").strip() or str(_VENV_REL)
    env["UV_PROJECT_ENVIRONMENT"] = rel_venv.replace("\\", "/")
    subprocess.run(
        [_uv_bin(), "sync", "--extra", "custom_actions"],
        cwd=repo_root,
        env=env,
        check=True,
    )
    py = _venv_python(venv)
    subprocess.run(
        [_uv_bin(), "pip", "install", "-e", str(repo_root), "--python", str(py)],
        cwd=repo_root,
        check=True,
    )
    marker.write_text("ok\n", encoding="utf-8")
    print(f"custom_actions: venv ready at {venv}", file=sys.stderr, flush=True)
    return py


def ensure_code_model(repo_root: Path) -> None:
    """Download default $code GGUF into models/ if missing."""
    venv = custom_venv_dir(repo_root)
    marker = venv / _MODEL_MARKER
    if marker.is_file() and os.environ.get("AH_FORCE_CODE_MODEL_DOWNLOAD", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    py = ensure_venv(repo_root)
    model = os.environ.get("AH_CUSTOM_ACTIONS_CODE_MODEL", "default")
    script = (
        "import os\n"
        "os.environ.setdefault('AH_MODEL_UPSTREAM_FALLBACK', '1')\n"
        "from externals.code.model_paths import ensure_model, resolve_profile_key\n"
        f"ensure_model(key=resolve_profile_key({model!r}))\n"
    )
    subprocess.run(
        [str(py), "-c", script],
        cwd=repo_root,
        env=_subprocess_env(repo_root),
        check=True,
    )
    marker.write_text("ok\n", encoding="utf-8")


def parse_third_party_imports(source: str) -> list[str]:
    """Return sorted unique top-level module names from import statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: set[str] = set()
    stdlib = _stdlib_modules()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top and top not in stdlib and top not in _SKIP_MODULES:
                    found.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            mod = node.module or ""
            top = mod.split(".")[0]
            if top and top not in stdlib and top not in _SKIP_MODULES:
                found.add(top)
    return sorted(found)


def modules_to_pypi_packages(modules: list[str]) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for mod in modules:
        pkg = _MODULE_TO_PYPI.get(mod, mod)
        key = pkg.lower()
        if key not in seen:
            seen.add(key)
            packages.append(pkg)
    return packages


_PKG_IMPORT_PROBE: dict[str, str] = {
    "Pillow": "PIL",
    "opencv-python-headless": "cv2",
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "beautifulsoup4": "bs4",
    "python-dotenv": "dotenv",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
}


def _probe_import_name(pypi_package: str) -> str:
    base = pypi_package.split("[")[0].split("==")[0].split(">=")[0].strip()
    return _PKG_IMPORT_PROBE.get(base, base)


def _package_importable(py: Path, pypi_package: str) -> bool:
    mod = _probe_import_name(pypi_package)
    r = subprocess.run(
        [str(py), "-c", f"import {mod}"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def install_pypi_packages(repo_root: Path, packages: list[str]) -> list[str]:
    """uv pip install missing packages into the custom-actions venv."""
    if not packages:
        return []
    py = ensure_venv(repo_root)
    installed: list[str] = []
    for pkg in packages:
        if _package_importable(py, pkg):
            continue
        print(
            f"custom_actions: installing {pkg!r} (from imports)",
            file=sys.stderr,
            flush=True,
        )
        subprocess.run(
            [_uv_bin(), "pip", "install", pkg, "--python", str(py)],
            cwd=repo_root,
            check=True,
        )
        installed.append(pkg)
    return installed


def sync_imports_for_code(
    repo_root: Path,
    code: str,
    meta_path: Path,
) -> list[str]:
    """Parse imports in code, install new PyPI deps, update meta.json."""
    modules = parse_third_party_imports(code)
    packages = modules_to_pypi_packages(modules)
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    prev = set(meta.get("pypi_packages") or [])
    new_pkgs = [p for p in packages if p not in prev]
    if new_pkgs:
        install_pypi_packages(repo_root, new_pkgs)
    meta["imports"] = modules
    meta["pypi_packages"] = packages
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return packages


def ensure_custom_actions_env(repo_root: Path) -> Path:
    """Venv + default code model (for &action codegen)."""
    py = ensure_venv(repo_root)
    ensure_code_model(repo_root)
    return py
