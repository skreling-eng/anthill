"""HTTP client for ACE-Step API server (acestep-api / release_task workflow)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _api_base() -> str:
    return os.environ.get("ACESTEP_API_URL", "http://127.0.0.1:8001").rstrip("/")


def _request(method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
    url = f"{_api_base()}{path}"
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("ACESTEP_API_KEY", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        body = dict(payload)
        if token and "ai_token" not in body:
            body["ai_token"] = token
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ACE-Step API {method} {path} failed ({exc.code}): {detail}") from exc
    parsed = json.loads(raw)
    if parsed.get("code") != 200:
        raise RuntimeError(f"ACE-Step API error: {parsed.get('error') or parsed}")
    return parsed.get("data", parsed)


def release_task(
    *,
    caption: str,
    lyrics: str,
    model: str | None = None,
    duration: float | None = None,
    seed: int | None = None,
    audio_format: str = "wav",
) -> str:
    payload: dict[str, Any] = {
        "prompt": caption,
        "lyrics": lyrics or "[Instrumental]",
        "thinking": False,
        "task_type": "text2music",
        "use_format": False,
        "audio_format": audio_format,
        "batch_size": 1,
    }
    if model:
        payload["model"] = model
    if duration is not None and duration > 0:
        payload["audio_duration"] = duration
    if seed is not None and seed >= 0:
        payload["use_random_seed"] = False
        payload["seed"] = seed
    data = _request("POST", "/release_task", payload)
    task_id = data.get("task_id") if isinstance(data, dict) else None
    if not task_id:
        raise RuntimeError(f"ACE-Step API returned no task_id: {data!r}")
    return str(task_id)


def wait_for_task(task_id: str, *, poll_sec: float = 2.0, timeout_sec: float = 1800.0) -> list[dict]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        rows = _request("POST", "/query_result", {"task_id_list": [task_id]})
        if not rows:
            time.sleep(poll_sec)
            continue
        row = rows[0]
        status = row.get("status")
        if status == 1:
            result_raw = row.get("result") or "[]"
            items = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
            return items if isinstance(items, list) else [items]
        if status == 2:
            raise RuntimeError(f"ACE-Step task failed: {row!r}")
        time.sleep(poll_sec)
    raise TimeoutError(f"ACE-Step task {task_id} timed out after {timeout_sec}s")


def download_audio(file_url: str, dest: Path) -> Path:
    if file_url.startswith("http"):
        url = file_url
    else:
        url = f"{_api_base()}{file_url}" if file_url.startswith("/") else f"{_api_base()}/{file_url}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def generate_via_api(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    model: str | None = None,
    duration: float | None = None,
    seed: int | None = None,
) -> Path:
    task_id = release_task(
        caption=caption,
        lyrics=lyrics,
        model=model,
        duration=duration,
        seed=seed,
        audio_format=output_path.suffix.lstrip(".") or "wav",
    )
    items = wait_for_task(task_id)
    if not items:
        raise RuntimeError("ACE-Step task returned no audio files")
    file_ref = items[0].get("file")
    if not file_ref:
        raise RuntimeError(f"ACE-Step result missing file URL: {items[0]!r}")
    return download_audio(str(file_ref), output_path)
