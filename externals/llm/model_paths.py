"""Resolve GGUF model files under models/ (shared search roots with image models)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import auto_download_enabled, require_models_file, resolve_models_file
from externals.image.model_paths import models_roots, resolve_model_path

_GEMMA_DEFAULT_DIR = "llm/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS"


def _gguf_from_dir(directory: Path) -> str | None:
    """Resolve model.gguf inside a model directory (optionally via model.yml)."""
    model_gguf = directory / "model.gguf"
    if model_gguf.is_file():
        return str(model_gguf.resolve())

    yml = directory / "model.yml"
    if yml.is_file():
        for line in yml.read_text(encoding="utf-8").splitlines():
            if line.startswith("model_path:"):
                rel = line.split(":", 1)[1].strip()
                local = directory / Path(rel.replace("\\", "/")).name
                if local.is_file():
                    return str(local.resolve())
                try:
                    found = resolve_model_path(rel)
                    if Path(found).is_file():
                        return str(Path(found).resolve())
                except OSError:
                    pass
                break

    for hit in sorted(directory.glob("*.gguf")):
        if "mmproj" not in hit.name.lower():
            return str(hit.resolve())
    return None


def resolve_gguf_path(ref: str) -> str:
    """
    Find a .gguf file by name or path.

    Searches MODELS_PATH and project models/ (including subfolders).
  """
    if not ref:
        raise FileNotFoundError("No GGUF model reference given")

    rel = ref.replace("\\", "/")
    if rel.lower().endswith(".gguf") and "/" in rel:
        found = resolve_models_file(rel)
        if found is not None:
            return str(found)
        if auto_download_enabled():
            try:
                return str(require_models_file(rel, label="$llm"))
            except FileNotFoundError:
                pass

    resolved = resolve_model_path(ref)
    path = Path(resolved)
    if path.is_dir():
        found = _gguf_from_dir(path)
        if found:
            return found
    if path.is_file() and path.suffix.lower() == ".gguf":
        return str(path.resolve())
    if path.suffix.lower() == ".gguf" and path.parent.is_dir():
        found = _gguf_from_dir(path.parent)
        if found:
            return found

    target = Path(ref).name
    if not target.lower().endswith(".gguf"):
        target = f"{target}.gguf"

    for root in models_roots():
        if not root.is_dir():
            continue
        direct = root / target
        if direct.is_file():
            return str(direct.resolve())
        try:
            for hit in root.rglob(target):
                if hit.is_file() and hit.suffix.lower() == ".gguf":
                    return str(hit.resolve())
        except OSError:
            continue

    raise FileNotFoundError(
        f"GGUF model not found: {ref!r} (searched under {', '.join(map(str, models_roots()))})"
    )
