"""Long-lived warm subprocess pool for GPU-heavy $ externals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


@dataclass(frozen=True)
class WarmWorkerConfig:
    """Configuration for a warm GPU worker subprocess."""

    name: str
    worker_module: str

    def env_key(self, suffix: str) -> str:
        return f"AH_{self.name.upper().replace('-', '_')}_{suffix}"


def worker_enabled(config: WarmWorkerConfig) -> bool:
    raw = os.environ.get(config.env_key("WORKER"), "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def default_comfy_worker_cmd(config: WarmWorkerConfig) -> list[str]:
    from externals.comfy_inprocess.bootstrap import resolve_comfy_python

    # Prefer Anthill isolated venv (e.g. .venvs/comfy-wan for image2video) over ComfyUI install.
    custom = external_python(config.name)
    if custom:
        return [custom, "-m", config.worker_module]
    isolated = venv_python(config.name, None)
    if isolated:
        return [isolated, "-m", config.worker_module]
    comfy_py = resolve_comfy_python()
    if comfy_py is not None:
        return [str(comfy_py), "-m", config.worker_module]
    if not _use_uv():
        return [sys.executable, "-m", config.worker_module]
    uv_bin = shutil.which(os.environ.get("UV", "uv")) or "uv"
    cmd = [uv_bin, "run"]
    for extra in uv_extras_for(config.name, None):
        cmd.extend(["--extra", extra])
    cmd.extend(["python", "-m", config.worker_module])
    return cmd


class WarmWorkerPool:
    """One warm subprocess per config; keeps models loaded between jobs."""

    def __init__(
        self,
        config: WarmWorkerConfig,
        *,
        build_cmd: Callable[[], list[str]] | None = None,
    ) -> None:
        self.config = config
        self._build_cmd = build_cmd or (lambda: default_comfy_worker_cmd(config))
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stdout_lock = threading.Lock()

    def enabled(self) -> bool:
        return worker_enabled(self.config)

    def terminate(self) -> None:
        proc = self._proc
        self._proc = None
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

    def _read_line(self, proc: subprocess.Popen, timeout: float | None) -> dict:
        if proc.stdout is None:
            raise RuntimeError(f"{self.config.name} worker stdout is closed")
        deadline = time.monotonic() + timeout if timeout is not None else None
        label = self.config.name
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{label} worker exited with code {proc.returncode}")
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"{label} worker response timed out")
            with self._stdout_lock:
                line = proc.stdout.readline()
            if not line:
                time.sleep(0.02)
                continue
            stripped = line.strip()
            if not stripped.startswith("{"):
                if stripped:
                    print(stripped, file=sys.stderr, flush=True)
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                print(stripped, file=sys.stderr, flush=True)
                continue

    def _ensure(self, ctx: ExternalContext) -> subprocess.Popen:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            return proc

        self.terminate()
        cmd = self._build_cmd()
        print(f"${self.config.name}: starting warm worker ({cmd[0]})", flush=True)
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
        startup = float(os.environ.get(self.config.env_key("WORKER_STARTUP"), "600"))
        ready = self._read_line(proc, startup)
        if ready.get("status") != "ready":
            proc.kill()
            raise RuntimeError(f"{self.config.name} worker bad startup: {ready}")
        self._proc = proc
        return proc

    def run(self, ctx: ExternalContext, inp: ExternalInput):
        from ahlib.ah_runtime import ArrayBundle, RuntimeCancelled

        write_invoke(ctx.op_dir, inp)
        op_dir = ctx.op_dir.resolve()
        timeout_raw = os.environ.get("AH_EXTERNAL_TIMEOUT", "").strip()
        timeout = float(timeout_raw) if timeout_raw else None
        label = self.config.name

        with self._lock:
            proc = self._ensure(ctx)
            payload = {
                "op_dir": str(op_dir),
                "session_base_dir": str(ctx.base_dir.resolve()),
            }
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                raise RuntimeCancelled(f"${label} cancelled")

            try:
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
            except OSError as exc:
                self.terminate()
                raise RuntimeError(f"{label} worker stdin closed") from exc

            started = time.monotonic()
            while True:
                if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                    self.terminate()
                    raise RuntimeCancelled(f"${label} cancelled")
                if timeout is not None and time.monotonic() - started > timeout:
                    self.terminate()
                    raise TimeoutError(f"${label} worker timed out after {timeout}s")
                try:
                    msg = self._read_line(proc, timeout=2.0)
                except TimeoutError:
                    continue
                status = msg.get("status")
                if status == "ok" and msg.get("op_dir") == str(op_dir):
                    break
                if status == "error" and msg.get("op_dir") == str(op_dir):
                    err_path = op_dir / "error.txt"
                    detail = (
                        err_path.read_text(encoding="utf-8")
                        if err_path.is_file()
                        else msg.get("error", "")
                    )
                    if "OutOfMemoryError" in detail or "out of memory" in detail.lower():
                        self.terminate()
                    raise RuntimeError(f"${label} worker failed:\n{detail}")
                if status not in ("ok", "error"):
                    continue

        out_path = op_dir / "output.json"
        if not out_path.is_file():
            raise FileNotFoundError(f"{label} worker did not write {out_path}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return ArrayBundle.from_dict(data)
