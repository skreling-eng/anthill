"""In-process ACE-Step via the official ace-step package (PyTorch / *.safetensors)."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_handlers: dict[str, object] = {}


def native_available() -> bool:
    try:
        import acestep  # noqa: F401

        return True
    except ImportError:
        return False


def _project_root() -> str:
    raw = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if raw:
        return raw
    return str(Path(__file__).resolve().parents[2])


def _checkpoints_dir(models_dir: Path) -> Path:
    from externals.music.models_env import configure_models_environment

    configure_models_environment()
    raw = os.environ.get("ACESTEP_CHECKPOINTS_DIR", "").strip()
    if raw:
        return Path(raw)
    return models_dir


def _ensure_handlers(*, config_path: str, checkpoints_dir: Path):
    cache_key = f"{checkpoints_dir}|{config_path}"
    with _lock:
        if _handlers.get("cache_key") == cache_key and _handlers.get("ready"):
            return _handlers["dit"], _handlers["llm"]

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        llm = LLMHandler()
        device = os.environ.get("ACESTEP_DEVICE", "auto")
        os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", str(checkpoints_dir))
        msg, ok = dit.initialize_service(
            project_root=_project_root(),
            config_path=config_path,
            device=device,
        )
        if not ok:
            raise RuntimeError(msg or "ACE-Step initialize_service failed")

        _handlers.clear()
        _handlers.update(
            dit=dit,
            llm=llm,
            cache_key=cache_key,
            ready=True,
        )
        return dit, llm


def generate_via_native(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    config_path: str,
    checkpoints_dir: Path,
    duration: float,
    seed: int,
    steps: int,
    audio_format: str = "wav",
    param_defaults: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    ckpt = _checkpoints_dir(checkpoints_dir)
    print(f"$music native PyTorch checkpoints={ckpt} config={config_path}")

    dit, llm = _ensure_handlers(config_path=config_path, checkpoints_dir=ckpt)
    fmt = audio_format.lstrip(".").lower() or "wav"
    if fmt not in ("mp3", "wav", "flac", "wav32", "opus", "aac"):
        fmt = "wav"

    defaults = dict(param_defaults or {})
    defaults.setdefault("duration", duration)
    if steps > 0:
        defaults["inference_steps"] = steps

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics or "[Instrumental]",
        duration=float(defaults.pop("duration", duration)),
        inference_steps=int(defaults.pop("inference_steps", steps or 8)),
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        seed=seed if seed >= 0 else -1,
    )
    for key, value in {**defaults, **(extras or {})}.items():
        if hasattr(params, key) and key not in ("caption", "lyrics", "task_type"):
            setattr(params, key, value)

    config = GenerationConfig(
        batch_size=1,
        use_random_seed=seed < 0,
        seeds=[seed] if seed >= 0 else None,
        audio_format=fmt,
    )
    save_dir = output_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    result = generate_music(
        dit,
        llm,
        params,
        config,
        save_dir=str(save_dir),
    )
    if not result.success or not result.audios:
        raise RuntimeError(result.error or "ACE-Step generation failed")

    src = Path(result.audios[0].get("path") or "")
    if not src.is_file():
        raise RuntimeError("ACE-Step did not write an audio file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(src), str(output_path))
    return output_path
