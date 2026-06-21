"""BS-RoFormer separation via python-audio-separator (PyTorch)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from externals.music_separation.models import ROFORMER_MODELS_DIR, normalize_stem_name


def require_audio_separator() -> None:
    try:
        import audio_separator  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$music_separation / $split_song need audio-separator in .venvs/media.\n"
            "  powershell -File tools\\setup_external_venvs.ps1\n"
            "  Or: UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media --extra music_separation\n"
            "Test stub: AH_EMULATE_MUSIC_SEPARATION=1 or AH_EMULATE_SPLIT_SONG=1"
        ) from exc


def _stem_from_output_path(path: Path, expected: tuple[str, ...]) -> str:
    # Prefer explicit stem in parentheses: track_(Vocals)_model...
    match = re.search(r"\(([^)]+)\)", path.stem)
    if match:
        stem = normalize_stem_name(match.group(1))
        if stem in expected:
            return stem

    name = path.stem.lower()
    for stem in expected:
        if re.search(rf"(?<![a-z]){re.escape(stem)}(?![a-z])", name):
            return stem
    for token in re.findall(r"[a-z]+", name):
        stem = normalize_stem_name(token)
        if stem in expected:
            return stem
    raise RuntimeError(
        f"$music_separation: could not infer stem from output file {path.name!r}; "
        f"expected one of {expected}"
    )


def _resolve_output_path(raw: str, out_dir: Path) -> Path:
    """audio-separator returns basenames; files live under output_dir."""
    out_dir = out_dir.resolve()
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    for candidate in (out_dir / path.name, out_dir / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"$music_separation: RoFormer output file not found: {raw!r} "
        f"(looked under {out_dir})"
    )


def separate_file(
    audio_path: Path,
    *,
    model_filename: str,
    expected_stems: tuple[str, ...],
    work_dir: Path,
    model_file_dir: Path | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Path]:
    """Run audio-separator; return ``{stem: wav_path}``."""
    import logging as py_logging

    from audio_separator.separator import Separator

    require_audio_separator()
    models_dir = model_file_dir or ROFORMER_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = models_dir / model_filename
    if checkpoint is not None and checkpoint.is_file():
        if checkpoint.resolve() != target_ckpt.resolve():
            shutil.copy2(checkpoint, target_ckpt)

    out_dir = work_dir / "roformer_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        log_level=py_logging.WARNING,
        output_dir=str(out_dir),
        output_format="WAV",
        model_file_dir=str(models_dir),
        use_autocast=True,
    )
    separator.load_model(model_filename=model_filename)
    custom_names = {stem: stem for stem in expected_stems}
    output_files = separator.separate(str(audio_path), custom_output_names=custom_names)

    stems: dict[str, Path] = {}
    for raw in output_files:
        path = _resolve_output_path(raw, out_dir)
        stem = _stem_from_output_path(path, expected_stems)
        stems[stem] = path

    missing = [s for s in expected_stems if s not in stems]
    if missing:
        raise RuntimeError(
            f"$music_separation: RoFormer model {model_filename!r} did not produce "
            f"stems {missing}. Got: {sorted(stems)}"
        )
    return stems
