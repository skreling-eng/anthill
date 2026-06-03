"""User GGUF models under models/llm_user/<name>/ (not uploaded to anthill)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

from externals.image.model_paths import models_roots
from externals.llm.gguf_llm import GgufLlm

LLM_USER_DIR = "llm_user"
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_model_name(name: str) -> str:
    """Keep only ASCII letters, digits, underscore, hyphen."""
    cleaned = _NAME_RE.sub("", (name or "").strip())
    if not cleaned:
        raise ValueError(
            "$add_gguf_llm_model: name must contain at least one of [-A-Za-z0-9_]"
        )
    return cleaned


def _sanitize_filename(name: str) -> str:
    base = Path(unquote(name)).name
    if base.lower().endswith(".gguf"):
        stem = base[:-5]
    else:
        stem = base
    stem = _FILENAME_RE.sub("", stem) or "model"
    return f"{stem}.gguf"


def user_model_dir(model_name: str) -> Path:
    safe = sanitize_model_name(model_name)
    for root in models_roots():
        candidate = root / LLM_USER_DIR / safe
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / LLM_USER_DIR / safe


def _pick_gguf_in_dir(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    hits = [
        p
        for p in sorted(directory.glob("*.gguf"))
        if p.is_file() and "mmproj" not in p.name.lower()
    ]
    return hits[0] if hits else None


def user_gguf_path(model_name: str) -> Path | None:
    """Return existing GGUF for a user model name, if present."""
    safe = sanitize_model_name(model_name)
    for root in models_roots():
        directory = root / LLM_USER_DIR / safe
        found = _pick_gguf_in_dir(directory)
        if found is not None:
            return found
    return None


def user_llm_profile(model_name: str) -> GgufLlm | None:
    path = user_gguf_path(model_name)
    if path is None:
        return None
    safe = sanitize_model_name(model_name)
    return GgufLlm(
        safe,
        str(path.resolve()),
        n_ctx=8192,
        chat_format="chatml",
    )


def parse_gguf_source(link: str) -> tuple[str, str, str]:
    """
    Parse gguf= into (kind, repo_or_url, filename).

    kind: 'hf' | 'http'
    """
    raw = (link or "").strip()
    if not raw:
        raise ValueError("$add_gguf_llm_model: gguf= is required")

    if "::" in raw and not raw.startswith("http"):
        repo, _, filename = raw.partition("::")
        filename = _sanitize_filename(filename.strip() or "model.gguf")
        repo = repo.strip().strip("/")
        if not repo or "/" not in repo:
            raise ValueError(
                "$add_gguf_llm_model: use repo_id::filename.gguf "
                "(e.g. unsloth/Qwen3.6-35B-A3B-GGUF::Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; "
                "avoid *-MTP-GGUF repos until llama-cpp-python ships MTP support)"
            )
        return "hf", repo, filename

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if "huggingface.co" in host or host.endswith("hf.co"):
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 3:
            raise ValueError(
                "$add_gguf_llm_model: Hugging Face URL must include repo and file "
                "(…/resolve/main/file.gguf)"
            )
        if parts[0] in ("spaces", "datasets"):
            raise ValueError("$add_gguf_llm_model: model repo URL required, not spaces/datasets")
        repo_id = f"{parts[0]}/{parts[1]}"
        if parts[2] in ("resolve", "blob", "raw"):
            if len(parts) < 5:
                raise ValueError("$add_gguf_llm_model: missing filename in Hugging Face URL")
            filename = _sanitize_filename("/".join(parts[4:]))
            return "hf", repo_id, filename
        filename = _sanitize_filename("/".join(parts[2:]))
        return "hf", repo_id, filename

    if parsed.scheme in ("http", "https"):
        filename = _sanitize_filename(Path(parsed.path).name or "model.gguf")
        return "http", raw, filename

    if raw.lower().endswith(".gguf"):
        return "http", raw, _sanitize_filename(Path(raw).name)

    raise ValueError(
        "$add_gguf_llm_model: gguf= must be an http(s) URL, Hugging Face file URL, "
        "or repo_id::filename.gguf"
    )


def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "$add_gguf_llm_model needs huggingface-hub: uv pip install huggingface-hub"
        ) from exc

    print(
        f"$add_gguf_llm_model: downloading {filename} from {repo_id}",
        flush=True,
    )
    downloaded = Path(
        hf_hub_download(repo_id, filename, local_dir=str(dest.parent))
    )
    if downloaded.resolve() != dest.resolve():
        shutil.copy2(downloaded, dest)


def _download_http(url: str, dest: Path) -> None:
    import urllib.request

    print(f"$add_gguf_llm_model: downloading {url}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        shutil.move(str(tmp), dest)
    finally:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def ensure_user_gguf(model_name: str, gguf_link: str, *, emulate: bool = False) -> Path:
    """Download user GGUF into models/llm_user/<name>/<file>.gguf if missing."""
    safe = sanitize_model_name(model_name)
    existing = user_gguf_path(safe)
    if existing is not None:
        print(f"$add_gguf_llm_model: using existing {existing}", flush=True)
        return existing

    kind, repo_or_url, filename = parse_gguf_source(gguf_link)
    dest_dir = user_model_dir(safe)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if emulate:
        dest.write_bytes(b"[emulated gguf]\n")
        print(f"$add_gguf_llm_model: emulated placeholder at {dest}", flush=True)
        return dest

    if kind == "hf":
        _download_hf(repo_or_url, filename, dest)
    else:
        _download_http(repo_or_url, dest)

    if not dest.is_file():
        raise FileNotFoundError(f"$add_gguf_llm_model: download failed: {dest}")
    return dest
