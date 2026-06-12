"""Subprocess IPC for $ externals: input.json + invoke.json -> output.json."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from externals.api import ExternalContext, ExternalInput

_REPO_ROOT = Path(__file__).resolve().parent.parent

_active_procs_lock = threading.Lock()
_active_procs: list[subprocess.Popen] = []


def terminate_active_subprocesses() -> None:
    """Stop any $ external subprocess started by this process."""
    for name in ("image2image", "image2video"):
        try:
            mod = __import__(
                f"externals.{name}.worker_client",
                fromlist=["terminate_worker"],
            )
            mod.terminate_worker()
        except ImportError:
            pass
    with _active_procs_lock:
        procs = list(_active_procs)
        _active_procs.clear()
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


def release_gpu_resources(*, reason: str = "") -> None:
    """Stop GPU-holding worker/subprocesses and empty CUDA cache in this process."""
    raw = os.environ.get("AH_RELEASE_GPU_ON_RUN_END", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    suffix = f" ({reason})" if reason else ""
    print(f"$externals: releasing GPU resources{suffix}", flush=True)
    terminate_active_subprocesses()
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

# uv --extra passed when spawning subprocess externals (AH_UV_EXTRA_<name> overrides).
_DEFAULT_UV_EXTRAS: dict[str, list[str]] = {
    "image": ["media"],
    "image2image": ["media"],
    "check_image": ["media"],
    "image2video": ["media", "comfy-wan", "clip"],
    "image_clip": ["media"],
    "video_clip": ["media"],
    "clip": ["clip"],
    "music": ["music"],
    "music_separation": ["media", "music_separation"],
    "change_voice": [],
    "join_stems": ["media", "music_separation"],
    "text2speech": ["text2speech"],
    "voice_enhance": [],
    "llm": [],
    "code": [],
    "sound2text": ["sound2text"],
    "ocr": ["ocr"],
    "image2text": ["media"],
    "image2embedding": ["image2embedding"],
    "text2embedding": ["image2embedding"],
    "video2embedding": ["video2embedding"],
    "add_video_embedding_files": ["media", "split_video_fast"],
    "create_video_index": ["media", "video_index"],
    "search_local_video": ["media", "video_index", "split_video", "video2embedding"],
    "split_video": ["split_video"],
    "split_video_fast": ["split_video_fast"],
    "translate": ["media"],
    "audio_instruct": ["media"],
    "video_thumbnailer": ["video_thumbnailer"],
    "model_ah_train_lora": ["finetune"],
    "model_ah_merge_lora": ["finetune"],
    "face": ["media"],
    "face_enhancer": ["media"],
}

# Cheap I/O / bundle ops — avoid uv-run subprocess spawn per call (override with
# AH_EXTERNAL_SUBPROCESS=file,… if you need isolation).
_DEFAULT_INPROCESS = frozenset(
    {
        "file",
        "folder",
        "clear",
        "pass",
        "del_session",
        "list",
        "first_image",
        "input_json",
        "save",
        "output",
        "only",
        "select",
        "texts_to_prompts",
        "texts2prompts",
        "prompts_to_texts",
        "prompts2texts",
        "json2texts",
        "ah_code_examples",
        "model_ah_create_jsonl",
        "code",
    }
)

# Isolated venv dirs (see tools/setup_external_venvs.ps1). Used when the path exists.
_DEFAULT_VENVS: dict[str, str] = {
    "image": ".venvs/media",
    "image2image": ".venvs/media",
    "check_image": ".venvs/media",
    "image2video": ".venvs/comfy-wan",
    "image_clip": ".venvs/media",
    "video_clip": ".venvs/media",
    "clip": ".venvs/media",
    "music": ".venvs/music",
    "music_separation": ".venvs/media",
    "change_voice": ".venvs/change_voice",
    "join_stems": ".venvs/media",
    "text2speech": ".venvs/text2speech",
    "voice_enhance": ".venvs/voice_enhance",
    "image2text": ".venvs/media",
    "image2embedding": ".venvs/media",
    "text2embedding": ".venvs/media",
    "video2embedding": ".venvs/media",
    "add_video_embedding_files": ".venvs/media",
    "create_video_index": ".venvs/media",
    "search_local_video": ".venvs/media",
    "split_video": ".venvs/media",
    "split_video_fast": ".venvs/media",
    "translate": ".venvs/media",
    "audio_instruct": ".venvs/media",
    "face": ".venvs/media",
    "face_enhancer": ".venvs/media",
}


def write_invoke(op_dir: Path, inp: ExternalInput) -> None:
    payload = {
        "args": dict(inp.args),
        "prompt_text": inp.prompt_text,
        "repeat": inp.repeat,
        "arg_lists": {k: list(v) for k, v in inp.arg_lists.items()},
    }
    (op_dir / "invoke.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_invoke(op_dir: Path) -> dict:
    path = op_dir / "invoke.json"
    if not path.is_file():
        return {
            "args": {},
            "prompt_text": "",
            "repeat": 1,
            "arg_lists": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invoke.json must be an object: {path}")
    return data


def subprocess_enabled(name: str) -> bool:
    """Default: subprocess. Opt out with AH_EXTERNAL_INPROCESS or AH_EXTERNAL_SUBPROCESS=0."""
    inprocess = os.environ.get("AH_EXTERNAL_INPROCESS", "").strip().lower()
    if inprocess in ("1", "true", "yes", "on", "all", "*"):
        return False
    if inprocess and name in {p.strip() for p in inprocess.split(",") if p.strip()}:
        return False

    raw = os.environ.get("AH_EXTERNAL_SUBPROCESS", "").strip().lower()
    if raw in ("0", "false", "no", "off", "inprocess"):
        return False
    if raw in ("1", "true", "yes", "on", "all", "*"):
        return True
    if raw:
        return name in {p.strip() for p in raw.split(",") if p.strip()}
    return name not in _DEFAULT_INPROCESS


def _use_uv() -> bool:
    raw = os.environ.get("AH_EXTERNAL_UV", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def uv_extras_for(name: str, op_dir: Path | None = None) -> list[str]:
    key = "AH_UV_EXTRA_" + name.upper().replace("-", "_")
    override = os.environ.get(key, "").strip()
    if override:
        if override in ("0", "none", "-"):
            return []
        extras = [p.strip() for p in override.split(",") if p.strip()]
    elif os.environ.get("AH_UV_EXTRAS", "").strip():
        extras = [
            p.strip()
            for p in os.environ.get("AH_UV_EXTRAS", "").split(",")
            if p.strip()
        ]
    else:
        extras = list(_DEFAULT_UV_EXTRAS.get(name, []))
    return extras


def external_python(name: str) -> str | None:
    """Explicit interpreter path; when set, uv is not used."""
    key = "AH_EXTERNAL_PYTHON_" + name.upper().replace("-", "_")
    custom = os.environ.get(key, "").strip() or os.environ.get(
        "AH_EXTERNAL_PYTHON", ""
    ).strip()
    return custom or None


def venv_path_for(name: str, op_dir: Path | None = None) -> Path | None:
    """Directory for an isolated uv venv (AH_EXTERNAL_VENV_<name> or .venvs/* default)."""
    key = "AH_EXTERNAL_VENV_" + name.upper().replace("-", "_")
    raw = os.environ.get(key, "").strip() or _DEFAULT_VENVS.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve() if path.is_dir() else None


def venv_python(name: str, op_dir: Path | None = None) -> str | None:
    root = venv_path_for(name, op_dir)
    if root is None:
        return None
    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = root / rel
        if candidate.is_file():
            return str(candidate)
    return None


def require_external_venv(name: str) -> None:
    """Raise when a configured isolated venv is missing (avoids broken .venv fallback)."""
    rel = _DEFAULT_VENVS.get(name, "").strip()
    if not rel:
        return
    path = Path(rel)
    if not path.is_absolute():
        path = _REPO_ROOT / rel
    if path.is_dir() and venv_python(name) is not None:
        return
    raise RuntimeError(
        f"$externals {name!r} needs isolated venv {rel} with torch/transformers/faiss.\n"
        f"  Run once: tools\\setup_external_venvs.ps1\n"
        f"  Or: UV_PROJECT_ENVIRONMENT={rel} uv sync --extra media --extra video_index"
    )


def build_runner_cmd(name: str, op_dir: Path) -> list[str]:
    """Command argv for externals.runner (isolated venv > uv run --extra > sys.executable)."""
    op = str(op_dir.resolve())
    custom = external_python(name)
    if custom:
        return [custom, "-m", "externals.runner", name, op]

    require_external_venv(name)
    isolated = venv_python(name, op_dir)
    if isolated:
        return [isolated, "-m", "externals.runner", name, op]

    if not _use_uv():
        return [sys.executable, "-m", "externals.runner", name, op]

    uv_bin = shutil.which(os.environ.get("UV", "uv")) or "uv"
    cmd = [uv_bin, "run"]
    for extra in uv_extras_for(name, op_dir):
        cmd.extend(["--extra", extra])
    cmd.extend(["python", "-m", "externals.runner", name, op])
    return cmd


def runner_cmd_display(cmd: list[str]) -> str:
    if len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] == "externals.runner":
        return cmd[0]
    parts: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == "--extra" and i + 1 < len(cmd):
            parts.append(f"--extra {cmd[i + 1]}")
            i += 2
            continue
        if cmd[i].endswith(("uv", "uv.exe")) or cmd[i] == "uv":
            parts.append("uv run")
            i += 1
            continue
        if cmd[i] == "run":
            i += 1
            continue
        if cmd[i] == "python":
            i += 1
            continue
        if "-m" in cmd[i:] and "externals.runner" in cmd:
            break
        i += 1
    return " ".join(parts) if parts else " ".join(cmd[:4])


def _isolated_subprocess(name: str, op_dir: Path | None) -> bool:
    return (
        name in _DEFAULT_VENVS
        and venv_python(name, op_dir) is not None
        and not external_python(name)
    )


def _subprocess_env(ctx: ExternalContext, name: str | None = None) -> dict[str, str]:
    from externals.bootstrap import load_dotenv

    load_dotenv()
    env = os.environ.copy()
    env["AH_SESSION_BASE_DIR"] = str(ctx.base_dir.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    # expandable_segments helps VRAM on Linux; PyTorch warns on Windows (unsupported).
    if os.name != "nt" and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    root = str(_REPO_ROOT)
    isolated = bool(name and _isolated_subprocess(name, ctx.op_dir))
    if isolated:
        # Parent uv/.venv PYTHONPATH must not shadow the isolated venv site-packages.
        env.pop("VIRTUAL_ENV", None)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = root
    else:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    return env


def run_external_subprocess(
    name: str, ctx: ExternalContext, inp: ExternalInput
) -> "ArrayBundle":
    from ahlib.ah_runtime import ArrayBundle, RuntimeCancelled

    write_invoke(ctx.op_dir, inp)
    op_dir = ctx.op_dir.resolve()
    cmd = build_runner_cmd(name, op_dir)
    timeout_raw = os.environ.get("AH_EXTERNAL_TIMEOUT", "").strip()
    timeout = float(timeout_raw) if timeout_raw else None

    print(
        f"$externals: subprocess {name!r} via {runner_cmd_display(cmd)}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        env=_subprocess_env(ctx, name),
    )
    with _active_procs_lock:
        _active_procs.append(proc)
    started = time.monotonic()
    try:
        while proc.poll() is None:
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                raise RuntimeCancelled(f"$externals subprocess {name!r} cancelled")
            if timeout is not None and time.monotonic() - started > timeout:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                raise TimeoutError(
                    f"$externals subprocess {name!r} timed out after {timeout}s"
                )
            time.sleep(0.1)
        if proc.returncode != 0:
            err_path = op_dir / "error.txt"
            detail = ""
            if err_path.is_file():
                lines = [
                    ln
                    for ln in err_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if ln.strip()
                ]
                if lines:
                    detail = "\n".join(lines[-20:])
            msg = f"$externals subprocess {name!r} failed (exit {proc.returncode})"
            if detail:
                msg = f"{msg}:\n{detail}"
            raise RuntimeError(msg) from None
    finally:
        with _active_procs_lock:
            try:
                _active_procs.remove(proc)
            except ValueError:
                pass

    out_path = op_dir / "output.json"
    if not out_path.is_file():
        raise FileNotFoundError(
            f"$externals subprocess {name!r} did not write {out_path}"
        )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return ArrayBundle.from_dict(data)
