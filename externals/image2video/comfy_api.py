"""Run $image2video via a running ComfyUI server (same speed as your Comfy workflow).

Requires a workflow exported in *API format* (Comfy UI: Save / Export API format).
Set COMFYUI_WORKFLOW to that JSON path. Comfy must be running (default http://127.0.0.1:8188).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _comfy_base() -> str:
    return os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def _http_json(method: str, path: str, payload: dict | None = None, *, timeout: float = 120) -> Any:
    url = f"{_comfy_base()}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI {method} {path} failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach ComfyUI at {_comfy_base()} — start ComfyUI desktop or "
            f"`python main.py --listen` in your Comfy install. ({exc})"
        ) from exc
    if not body:
        return {}
    return json.loads(body)


def _workflow_path() -> Path:
    raw = os.environ.get("COMFYUI_WORKFLOW", "").strip()
    if not raw:
        raise ValueError(
            "COMFYUI_WORKFLOW is not set. In ComfyUI, build your MEGA I2V graph, "
            "then export API-format JSON and set COMFYUI_WORKFLOW to that file."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COMFYUI_WORKFLOW not found: {path}")
    return path


def _load_workflow() -> dict[str, Any]:
    data = json.loads(_workflow_path().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("COMFYUI_WORKFLOW must be a JSON object (API export format)")
    # Some exports wrap under "prompt"
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data


def _upload_image(image_path: Path) -> str:
    """Upload to Comfy input folder; returns the image name Comfy expects in LoadImage."""
    from externals.comfy.client import ComfyClient

    return ComfyClient(_comfy_base()).upload_image(image_path)


def _patch_workflow(
    workflow: dict[str, Any],
    *,
    image_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int | None,
    guidance: float | None,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    """Best-effort patch of API workflow nodes (titles optional, matched by class_type)."""
    wf = json.loads(json.dumps(workflow))
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        meta = node.get("_meta", {}) or {}
        title = (meta.get("title") or "").lower()

        if cls == "LoadImage":
            inputs["image"] = image_name
            continue

        if cls in ("CLIPTextEncode", "WanVideoTextEncode", "WanVideoTextEncodeSingle"):
            if "negative" in title or "neg" in title:
                if negative_prompt:
                    inputs["text"] = negative_prompt
            elif "positive" in title or "prompt" in title or not negative_prompt:
                inputs["text"] = prompt
            elif "text" not in inputs or not str(inputs.get("text", "")).strip():
                inputs["text"] = prompt
            continue

        if cls in ("KSampler", "KSamplerAdvanced"):
            if seed:
                inputs["seed"] = seed
            if steps is not None:
                inputs["steps"] = steps
            if guidance is not None:
                for key in ("cfg", "guidance", "cfg_scale"):
                    if key in inputs or key in node.get("inputs", {}):
                        inputs[key] = guidance
            continue

        if "WanVideoSampler" in cls or cls.endswith("Sampler"):
            if seed:
                inputs["seed"] = seed
            if steps is not None and "steps" in inputs:
                inputs["steps"] = steps
            if guidance is not None:
                for key in ("cfg", "guidance", "cfg_scale", "guidance_scale"):
                    if key in inputs:
                        inputs[key] = guidance

        if width and height and cls in ("EmptyLatentImage", "WanVideoEmptyEmbeds"):
            for wk, hk in (("width", "height"), ("video_width", "video_height")):
                if wk in inputs and hk in inputs:
                    inputs[wk] = width
                    inputs[hk] = height

    return wf


def _wait_history(prompt_id: str, *, timeout_s: float = 7200) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = _http_json("GET", f"/history/{prompt_id}", timeout=30)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"ComfyUI workflow error: {msgs}")
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {timeout_s}s")


def _first_output_video(history_entry: dict) -> tuple[str, str]:
    outputs = history_entry.get("outputs", {})
    for node_out in outputs.values():
        for key in ("gifs", "videos", "images"):
            items = node_out.get(key) or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or item.get("name")
                subfolder = item.get("subfolder", "")
                ftype = item.get("type", "output")
                if filename and (filename.endswith(".mp4") or key in ("gifs", "videos")):
                    return filename, subfolder if subfolder else ftype
    raise RuntimeError("ComfyUI finished but no video output found in history")


def _download_output(filename: str, subfolder: str, dest: Path) -> Path:
    qs = urlencode({"filename": filename, "subfolder": subfolder, "type": "output"})
    url = f"{_comfy_base()}/view?{qs}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


def generate_via_comfy(
    *,
    image_path: Path,
    prompt: str,
    output_path: Path,
    negative_prompt: str = "",
    seed: int = 0,
    steps: int | None = None,
    guidance_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    print("$image2video: backend=comfy (ComfyUI API)", flush=True)
    workflow = _load_workflow()
    image_name = _upload_image(image_path)
    workflow = _patch_workflow(
        workflow,
        image_name=image_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        guidance=guidance_scale,
        width=width,
        height=height,
    )
    client_id = str(uuid.uuid4())
    queued = _http_json(
        "POST",
        "/prompt",
        {"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt returned no prompt_id: {queued}")
    print(f"$image2video: ComfyUI queued prompt_id={prompt_id}", flush=True)
    history = _wait_history(prompt_id)
    filename, subfolder = _first_output_video(history)
    return _download_output(filename, subfolder, output_path)
