"""Locate or download llama.cpp HF→GGUF conversion tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / ".cache" / "llama.cpp"
_LLama_CPP_ZIP = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip"

# llama-quantize names -> convert_hf_to_gguf --outtype when quantize binary missing
_PYTHON_OUTTYPE: dict[str, str] = {
    "Q8_0": "q8_0",
    "Q8": "q8_0",
    "F16": "f16",
    "F32": "f32",
    "BF16": "bf16",
    "AUTO": "auto",
}


def _auto_download_enabled() -> bool:
    raw = os.environ.get("AH_LLAMA_CPP_AUTO_DOWNLOAD", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _cache_dir() -> Path:
    raw = os.environ.get("AH_LLAMA_CPP_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CACHE.resolve()


def _find_convert_script(root: Path) -> Path | None:
    for rel in (
        "convert_hf_to_gguf.py",
        "examples/convert_hf_to_gguf.py",
        "tools/convert_hf_to_gguf.py",
    ):
        path = root / rel
        if path.is_file():
            return path
    return None


def _find_quantize_bin(root: Path) -> Path | None:
    for rel in (
        "build/bin/llama-quantize",
        "build/bin/Release/llama-quantize.exe",
        "build/bin/llama-quantize.exe",
        "llama-quantize",
        "llama-quantize.exe",
    ):
        path = root / rel
        if path.is_file():
            return path
    found = shutil.which("llama-quantize")
    return Path(found) if found else None


def _extracted_roots(cache: Path) -> list[Path]:
    roots = [cache]
    if cache.is_dir():
        roots.extend(p for p in cache.iterdir() if p.is_dir())
    return roots


def _download_llama_cpp(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "llama.cpp-master.zip"
    print(f"Downloading llama.cpp conversion tools -> {cache}", flush=True)
    with urllib.request.urlopen(_LLama_CPP_ZIP, timeout=120) as resp:
        zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache)
    try:
        zip_path.unlink()
    except OSError:
        pass
    for root in _extracted_roots(cache):
        if _find_convert_script(root) is not None:
            print(f"llama.cpp tools ready: {root}", flush=True)
            return root
    raise RuntimeError(
        f"Downloaded llama.cpp to {cache} but convert_hf_to_gguf.py was not found."
    )


def ensure_llama_cpp_root() -> Path:
    """Return llama.cpp root with convert_hf_to_gguf.py (env, cache, or auto-download)."""
    raw = os.environ.get("AH_LLAMA_CPP", "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_dir() and _find_convert_script(path):
            return path
        raise RuntimeError(
            f"AH_LLAMA_CPP={path} does not contain convert_hf_to_gguf.py"
        )

    for candidate in (
        *_extracted_roots(_cache_dir()),
        Path.home() / "llama.cpp",
        _REPO_ROOT / "llama.cpp",
        Path("llama.cpp"),
    ):
        if candidate.is_dir() and _find_convert_script(candidate):
            return candidate.resolve()

    if not _auto_download_enabled():
        raise RuntimeError(
            "GGUF conversion needs llama.cpp. Set AH_LLAMA_CPP to your llama.cpp clone, "
            "or leave AH_LLAMA_CPP_AUTO_DOWNLOAD=1 (default) to download tools on first use."
        )
    return _download_llama_cpp(_cache_dir())


def python_outtype_for_quant(quant: str) -> str:
    """Best --outtype when llama-quantize is unavailable."""
    key = quant.strip().upper().replace("-", "_")
    if key in _PYTHON_OUTTYPE:
        return _PYTHON_OUTTYPE[key]
    # K-quants (Q4_K_M, etc.) — q8_0 is closest pure-Python option
    return "q8_0"


def run_hf_to_gguf(
    *,
    hf_dir: Path,
    out_gguf: Path,
    quant: str = "Q4_K_M",
) -> Path:
    """Convert HF folder to GGUF; uses llama-quantize when built, else --outtype."""
    root = ensure_llama_cpp_root()
    convert = _find_convert_script(root)
    if convert is None:
        raise RuntimeError(f"convert_hf_to_gguf.py not found under {root}")

    hf_dir = hf_dir.resolve()
    out_gguf = out_gguf.resolve()
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    quant_bin = _find_quantize_bin(root)
    env = os.environ.copy()
    # Prefer pip gguf if bundled gguf-py is incomplete
    env.setdefault("NO_LOCAL_GGUF", "1")

    if quant_bin is not None:
        f16 = out_gguf.with_name(out_gguf.stem + "-f16.gguf")
        print(f"Converting HF -> f16 GGUF: {convert}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(convert),
                str(hf_dir),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ],
            cwd=str(root),
            check=True,
            env=env,
        )
        if not f16.is_file():
            raise RuntimeError(f"Conversion failed: {f16} not created")
        print(f"Quantizing {f16} -> {out_gguf} ({quant})", flush=True)
        subprocess.run(
            [str(quant_bin), str(f16), str(out_gguf), quant],
            check=True,
        )
        if not out_gguf.is_file():
            raise RuntimeError(f"Quantization failed: {out_gguf} not created")
        if f16 != out_gguf and f16.is_file():
            try:
                f16.unlink()
            except OSError:
                pass
        print(f"GGUF saved: {out_gguf}", flush=True)
        return out_gguf

    outtype = python_outtype_for_quant(quant)
    if quant.strip().upper().replace("-", "_") not in _PYTHON_OUTTYPE:
        print(
            f"llama-quantize not found; writing {outtype} GGUF instead of {quant}. "
            "Build llama.cpp or set AH_LLAMA_CPP to a clone with llama-quantize for K-quants.",
            flush=True,
        )
    print(f"Converting HF -> GGUF ({outtype}): {convert}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(convert),
            str(hf_dir),
            "--outfile",
            str(out_gguf),
            "--outtype",
            outtype,
        ],
        cwd=str(root),
        check=True,
        env=env,
    )
    if not out_gguf.is_file():
        raise RuntimeError(f"Conversion failed: {out_gguf} not created")
    print(f"GGUF saved: {out_gguf}", flush=True)
    return out_gguf
