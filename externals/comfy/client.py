"""ComfyUI HTTP client: load workflow JSON, patch placeholders, queue, collect outputs."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PLACEHOLDER_IMAGE = "INPUT_IMAGE.png"
PLACEHOLDER_IMAGE_ALT = "INPUT_IMAGE"
PLACEHOLDER_PROMPT = "INPUT_PROMPT"
PLACEHOLDER_SOUND = "INPUT_SOUND"
PLACEHOLDER_SEED = "INPUT_SEED"
IMAGE_PLACEHOLDERS = (PLACEHOLDER_IMAGE, PLACEHOLDER_IMAGE_ALT)
SEED_MIN = 1
SEED_MAX = 200_000_000

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".gif"}
_SOUND_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def resolve_workflow_path(ref: str, *, base_dir: Path | None = None) -> Path:
    raw = Path(ref)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if base_dir is not None:
        candidates.append(base_dir / raw)
    candidates.extend(
        [
            Path.cwd() / raw,
            _REPO_ROOT / raw,
            _REPO_ROOT / "comfy_workflows" / raw,
            Path.cwd() / "comfy_workflows" / raw,
            _REPO_ROOT / "workflows" / raw,
            _REPO_ROOT / "comfy" / raw,
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"$comfy: workflow JSON not found: {ref!r} (tried {tried})")


def load_workflow_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"$comfy: workflow root must be a JSON object: {path}")
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data


def workflow_contains(obj: Any, token: str) -> bool:
    if isinstance(obj, dict):
        return any(workflow_contains(v, token) for v in obj.values())
    if isinstance(obj, list):
        return any(workflow_contains(v, token) for v in obj)
    if isinstance(obj, str):
        return token in obj
    return False


def workflow_contains_any(obj: Any, tokens: tuple[str, ...]) -> bool:
    return any(workflow_contains(obj, token) for token in tokens)


def patch_placeholders(obj: Any, replacements: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: patch_placeholders(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [patch_placeholders(v, replacements) for v in obj]
    if isinstance(obj, str):
        out = obj
        for token in sorted(replacements, key=len, reverse=True):
            if token in out:
                out = out.replace(token, replacements[token])
        return out
    return obj


def patch_seed_placeholder(obj: Any, seed: int) -> Any:
    """Replace INPUT_SEED with a scalar INT (Comfy KSampler seed widget).

    Templates may use \"seed\": \"INPUT_SEED\" or the mistaken export form
    \"seed\": [\"INPUT_SEED\", 0] — the latter must become a number, not
    [seed, 0], which Comfy reads as a link to node id \"seed\".
    """
    if isinstance(obj, dict):
        return {k: patch_seed_placeholder(v, seed) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and obj[0] == PLACEHOLDER_SEED:
            return seed
        return [patch_seed_placeholder(v, seed) for v in obj]
    if obj == PLACEHOLDER_SEED:
        return seed
    if isinstance(obj, str) and PLACEHOLDER_SEED in obj:
        if obj == PLACEHOLDER_SEED:
            return seed
        try:
            return int(obj.replace(PLACEHOLDER_SEED, str(seed)))
        except ValueError:
            return obj.replace(PLACEHOLDER_SEED, str(seed))
    return obj


class ComfyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def http_json(
        self, method: str, path: str, payload: dict | None = None, *, timeout: float = 120
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ComfyUI {method} {path} failed ({exc.code}): {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot reach ComfyUI at {self.base_url} — start ComfyUI "
                f"(`python main.py --listen --port …`). ({exc})"
            ) from exc
        if not body:
            return {}
        return json.loads(body)

    def upload_image(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"$comfy: image not found: {image_path}")
        if image_path.stat().st_size < 32:
            raise ValueError(
                f"$comfy: image file too small ({image_path.stat().st_size} bytes): "
                f"{image_path}. Use a real $image output, not an empty/emulated placeholder."
            )

        boundary = f"----Anthill{uuid.uuid4().hex}"
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        file_bytes = image_path.read_bytes()

        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(value.encode())
            parts.append(b"\r\n")

        add_field("overwrite", "true")
        add_field("type", "input")
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
        parts.append(file_bytes)
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        url = f"{self.base_url}/upload/image"
        req = Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ComfyUI image upload failed ({url}, {exc.code}): {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"ComfyUI image upload failed ({url}): {exc}. "
                "Is ComfyUI running on that host/port?"
            ) from exc
        name = result.get("name") or image_path.name
        sub = result.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def stage_input_file(self, source: Path, *, target_name: str | None = None) -> str:
        """Copy a file into Comfy's input folder (for audio / nodes without upload API)."""
        comfy_input = os.environ.get("COMFYUI_INPUT_DIR", "").strip()
        if not comfy_input:
            raise RuntimeError(
                "$comfy: workflow uses INPUT_SOUND but COMFYUI_INPUT_DIR is not set "
                "(path to ComfyUI/input)."
            )
        dest_dir = Path(comfy_input).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = target_name or source.name
        dest = dest_dir / name
        shutil.copy2(source, dest)
        return name

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        client_id = str(uuid.uuid4())
        queued = self.http_json(
            "POST",
            "/prompt",
            {"prompt": workflow, "client_id": client_id},
            timeout=60,
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt returned no prompt_id: {queued}")
        return str(prompt_id)

    def wait_history(self, prompt_id: str, *, timeout_s: float = 7200) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            hist = self.http_json("GET", f"/history/{prompt_id}", timeout=30)
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI workflow error: {msgs}")
            time.sleep(1.0)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not finish within {timeout_s}s"
        )

    def download_view(
        self, filename: str, subfolder: str, ftype: str, dest: Path
    ) -> Path:
        qs = urlencode(
            {"filename": filename, "subfolder": subfolder, "type": ftype}
        )
        url = f"{self.base_url}/view?{qs}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        return dest
