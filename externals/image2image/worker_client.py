"""Warm subprocess pool for $image2image (avoids reloading ~28GB each job)."""

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
from externals.invoke import (
    _REPO_ROOT,
    _subprocess_env,
    _use_uv,
    external_python,
    uv_extras_for,
    venv_python,
    write_invoke,
)

_worker_lock = threading.Lock()
_worker_proc: subprocess.Popen | None = None
_worker_stdout_lock = threading.Lock()


def worker_enabled() -> bool:
    raw = os.environ.get("AH_IMAGE2IMAGE_WORKER", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _worker_cmd() -> list[str]:
    from externals.image2image.comfy_bootstrap import resolve_comfy_python

    comfy_py = resolve_comfy_python()
    if comfy_py is not None:
        return [str(comfy_py), "-m", "externals.image2image.worker"]
    custom = external_python("image2image")
    if custom:
        return [custom, "-m", "externals.image2image.worker"]
    isolated = venv_python("image2image", None)
    if isolated:
        return [isolated, "-m", "externals.image2image.worker"]
    if not _use_uv():
        return [sys.executable, "-m", "externals.image2image.worker"]
    uv_bin = shutil.which(os.environ.get("UV", "uv")) or "uv"
    cmd = [uv_bin, "run"]
    for extra in uv_extras_for("image2image", None):
        cmd.extend(["--extra", extra])
    cmd.extend(["python", "-m", "externals.image2image.worker"])
    return cmd


def _terminate_worker() -> None:
    global _worker_proc
    proc = _worker_proc
    _worker_proc = None
    if proc is None:
        return
    if proc.poll() is None:
        try:
            if proc.stdin is not None:
                proc.stdin.write('{"cmd":"shutdown"}\n')
                proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _read_worker_line(proc: subprocess.Popen, timeout: float | None) -> dict:
    if proc.stdout is None:
        raise RuntimeError("image2image worker stdout is closed")
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"image2image worker exited with code {proc.returncode}"
            )
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("image2image worker response timed out")
        with _worker_stdout_lock:
            line = proc.stdout.readline()
        if not line:
            time.sleep(0.02)
            continue
        stripped = line.strip()
        if not stripped.startswith("{"):
            # Worker progress logs must use stderr; forward stray stdout lines.
            if stripped:
                print(stripped, file=sys.stderr, flush=True)
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            print(stripped, file=sys.stderr, flush=True)
            continue


def _ensure_worker(ctx: ExternalContext) -> subprocess.Popen:
    global _worker_proc
    proc = _worker_proc
    if proc is not None and proc.poll() is None:
        return proc

    _terminate_worker()
    cmd = _worker_cmd()
    print(
        f"$image2image: starting warm worker ({cmd[0]})",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        env=_subprocess_env(ctx),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    ready = _read_worker_line(
        proc,
        float(os.environ.get("AH_IMAGE2IMAGE_WORKER_STARTUP", "600")),
    )
    if ready.get("status") != "ready":
        proc.kill()
        raise RuntimeError(f"image2image worker bad startup: {ready}")
    _worker_proc = proc
    return proc


def run_via_worker(ctx: ExternalContext, inp: ExternalInput):
    from ahlib.ah_runtime import ArrayBundle, RuntimeCancelled

    write_invoke(ctx.op_dir, inp)
    op_dir = ctx.op_dir.resolve()
    timeout_raw = os.environ.get("AH_EXTERNAL_TIMEOUT", "").strip()
    timeout = float(timeout_raw) if timeout_raw else None

    with _worker_lock:
        proc = _ensure_worker(ctx)
        payload = {
            "op_dir": str(op_dir),
            "session_base_dir": str(ctx.base_dir.resolve()),
        }
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            raise RuntimeCancelled("$image2image cancelled")

        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            _terminate_worker()
            raise RuntimeError("image2image worker stdin closed") from exc

        started = time.monotonic()
        while True:
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                _terminate_worker()
                raise RuntimeCancelled("$image2image cancelled")
            if timeout is not None and time.monotonic() - started > timeout:
                _terminate_worker()
                raise TimeoutError(
                    f"$image2image worker timed out after {timeout}s"
                )
            try:
                msg = _read_worker_line(proc, timeout=2.0)
            except TimeoutError:
                continue
            status = msg.get("status")
            if status == "ok" and msg.get("op_dir") == str(op_dir):
                break
            if status == "error" and msg.get("op_dir") == str(op_dir):
                err_path = op_dir / "error.txt"
                detail = err_path.read_text(encoding="utf-8") if err_path.is_file() else msg.get("error", "")
                raise RuntimeError(f"$image2image worker failed:\n{detail}")
            if status not in ("ok", "error"):
                continue
            # stale line from a previous job; keep reading

    out_path = op_dir / "output.json"
    if not out_path.is_file():
        raise FileNotFoundError(f"image2image worker did not write {out_path}")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return ArrayBundle.from_dict(data)


def terminate_worker() -> None:
    with _worker_lock:
        _terminate_worker()
