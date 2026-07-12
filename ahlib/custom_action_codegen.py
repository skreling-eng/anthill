"""Generate, cache, validate, and load custom &action Python handlers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# First attempt + this many retries after validation failure.
_CODEGEN_MAX_RETRIES = 4


def _codegen_max_attempts() -> int:
    raw = os.environ.get("AH_CUSTOM_ACTION_MAX_ATTEMPTS", "").strip()
    if raw:
        return max(1, int(raw))
    return 1 + _CODEGEN_MAX_RETRIES


@dataclass
class _FailedAttempt:
    attempt: int
    code: str
    reason: str
    validator_response: str = ""

_GENERATION_SYSTEM = """You are a Python engineer implementing Anthill custom actions.

Write a single module that defines:

    def run(bundle: dict, base_dir: str, op_dir: str) -> dict:

INPUT — bundle dict (always present keys; lists may be empty):
  prompts, texts, images, sounds, videos, files, changes
  Each media key holds session-relative path strings (e.g. "3__image/images/0.png").
  changes: list of [content_type, operation, data] tuples (usually pass through unchanged).

OUTPUT — return a new bundle dict with the same keys. Values in media arrays are always
session-relative path LINK STRINGS (never PIL images, bytes, or file objects).
When the spec says "return an image" or "return texts", write the file and put the link
returned by save_* into the array — that is the Anthill output format.

OUTPUT FILES — use ahlib.custom_action_io helpers; they write under op_dir and RETURN the link
to put in bundle[] (never invent a separate path string):

  from ahlib.custom_action_io import (
      apply_db_gain, float_to_int16, save_wav, save_image, save_bytes, save_text,
  )

  link = save_wav(base_dir, op_dir, "louder_0.wav", sample_rate, data)  # → sounds[]
  link = save_image(base_dir, op_dir, "crop_0.png", pil_image)          # → images[]
  link = save_bytes(base_dir, op_dir, "files", "out.bin", raw_bytes)    # → files[]
  link = save_text(base_dir, op_dir, "prompts", "0.txt", prompt_text)    # → prompts[]

VIDEOS — use moviepy (ffmpeg on PATH). Do NOT use save_bytes for .mp4/.webm; that helper is for
files[] binary blobs only. Write under op_dir/videos/ and return a session-relative link string:

  from pathlib import Path
  from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip

  dest = Path(op_dir) / "videos" / "joined_0.mp4"
  dest.parent.mkdir(parents=True, exist_ok=True)
  final.write_videofile(str(dest), codec="libx264", audio_codec="aac", logger=None)
  link = dest.relative_to(Path(base_dir)).as_posix()   # → videos[]
  final.close()   # and close every clip you opened

MoviePy 2.x: subclipped(), with_audio(), with_fps(). MoviePy 1.x fallbacks: subclip(), set_audio().
Always close clips in finally blocks. For concat: concatenate_videoclips(clips, method="compose").

Read inputs with:  src = Path(base_dir) / link

ALLOWED IMPORTS:
  ahlib.custom_action_io, json, math, re, pathlib.Path, shutil, uuid
  PIL / Pillow for images; numpy / scipy.io.wavfile for audio when needed
  moviepy for video read/write/concat/mux (requires ffmpeg)

FORBIDDEN (validation will reject):
  subprocess, os.system, socket, urllib, requests, httpx, eval, exec, compile,
  __import__, open() on paths outside base_dir, deleting unrelated files, network, shells.

EXAMPLES:

1) Pass-through (no-op):
    def run(bundle, base_dir, op_dir):
        return {k: list(bundle.get(k, [])) for k in
                ("prompts","texts","images","sounds","videos","files","changes")}

2) Crop bottom 100px from each image:
    from pathlib import Path
    from PIL import Image
    from ahlib.custom_action_io import save_image
    def run(bundle, base_dir, op_dir):
        root = Path(base_dir)
        out = {k: list(bundle.get(k, [])) for k in
               ("prompts","texts","images","sounds","videos","files","changes")}
        new_images = []
        for link in bundle.get("images", []):
            im = Image.open(root / link).convert("RGB")
            w, h = im.size
            im = im.crop((0, 0, w, max(1, h - 100)))
            new_images.append(save_image(base_dir, op_dir, f"crop_{len(new_images)}.png", im))
        out["images"] = new_images
        return out

3) Louder sounds (+10 dB):
    from pathlib import Path
    from scipy.io import wavfile
    from ahlib.custom_action_io import apply_db_gain, float_to_int16, save_wav
    def run(bundle, base_dir, op_dir):
        root = Path(base_dir)
        out = {k: list(bundle.get(k, [])) for k in
               ("prompts","texts","images","sounds","videos","files","changes")}
        new_sounds = []
        for link in bundle.get("sounds", []):
            sr, data = wavfile.read(root / link)
            louder = float_to_int16(apply_db_gain(data, 10))
            new_sounds.append(save_wav(base_dir, op_dir, f"louder_{len(new_sounds)}.wav", sr, louder))
        out["sounds"] = new_sounds
        return out

4) Join videos[] into one MP4:
    from pathlib import Path
    from moviepy import VideoFileClip, concatenate_videoclips
    def run(bundle, base_dir, op_dir):
        root = Path(base_dir)
        out = {k: list(bundle.get(k, [])) for k in
               ("prompts","texts","images","sounds","videos","files","changes")}
        clips = []
        try:
            for link in bundle.get("videos", []):
                clips.append(VideoFileClip(str(root / link)))
            if not clips:
                return out
            final = concatenate_videoclips(clips, method="compose")
            clips.append(final)
            dest = Path(op_dir) / "videos" / "joined_0.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            final.write_videofile(str(dest), codec="libx264", audio_codec="aac", logger=None)
            out["videos"] = [dest.relative_to(root).as_posix()]
        finally:
            for clip in clips:
                clip.close()
        return out

5) Replace each video's audio with paired sounds[] (same length lists):
    from pathlib import Path
    from moviepy import AudioFileClip, VideoFileClip
    def run(bundle, base_dir, op_dir):
        root = Path(base_dir)
        out = {k: list(bundle.get(k, [])) for k in
               ("prompts","texts","images","sounds","videos","files","changes")}
        new_videos = []
        for video_link, sound_link in zip(bundle.get("videos", []), bundle.get("sounds", [])):
            video = VideoFileClip(str(root / video_link))
            audio = AudioFileClip(str(root / sound_link))
            try:
                end = min(video.duration, audio.duration)
                if hasattr(video, "subclipped"):
                    video = video.subclipped(0, end)
                else:
                    video = video.subclip(0, end)
                if hasattr(video, "with_audio"):
                    video = video.with_audio(audio)
                else:
                    video = video.set_audio(audio)
                dest = Path(op_dir) / "videos" / f"synced_{len(new_videos)}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                video.write_videofile(str(dest), codec="libx264", audio_codec="aac", logger=None)
                new_videos.append(dest.relative_to(root).as_posix())
            finally:
                video.close()
                audio.close()
        out["videos"] = new_videos
        return out

CRITICAL: append only the link string RETURNED by save_wav / save_image / save_bytes, or the
relative path string for moviepy outputs under op_dir/videos/.
Never build output paths by hand or call uuid.uuid4() for bundle links.

Follow the user's specification exactly. Output ONLY Python source (no markdown fences, no prose).
"""

_VALIDATION_SYSTEM = """You are a security reviewer for Anthill custom-action Python modules.

You receive:
1) The user's specification for what the action should do
2) The generated Python source

Reply with ONLY one JSON object (no markdown):
  {"ok": true}
or
  {"ok": false, "reason": "short explanation"}

ANTHILL DATA MODEL — required; do NOT reject correct code for misunderstanding this:
- bundle arrays (prompts, texts, images, sounds, videos, files) hold SESSION-RELATIVE PATH
  STRINGS (links like "3__crop/images/0.png"), never raw bytes, PIL Image objects, numpy arrays,
  or open file handles in the returned dict.
- When the spec says "return an image", "return the original image", "output a flipped image",
  "return texts", "join videos", "concatenate clips", etc., the handler MUST write files under
  op_dir (via save_image, save_wav, save_bytes, moviepy write_videofile, or equivalent) and put
  the RETURNED LINK STRING into the matching bundle array. That is the correct implementation —
  not a mismatch with the spec.
- save_image(base_dir, op_dir, filename, pil_image) writes the file AND returns the link;
  appending that link to out["images"] correctly "returns" the image in Anthill's format.
- For videos[], moviepy write_videofile to op_dir/videos/*.mp4 (or .webm) with a returned
  relative link is CORRECT. Do NOT reject moviepy/ffmpeg for video output. Do NOT require
  save_bytes for video files — save_bytes is for files[] only.
- Saving under op_dir and returning those links is REQUIRED and CORRECT. Never mark ok:false
  because the code returns paths instead of embedding image/text data in the bundle.

Mark ok:false if the code:
- Uses forbidden APIs (subprocess, os.system, eval, exec, compile, __import__, network libs)
- Reads/writes files outside base_dir (except under base_dir via Path(base_dir) / link)
- Deletes or overwrites unrelated session data
- Does something unrelated to the specification
- Lacks def run(bundle, base_dir, ...) -> dict (op_dir third parameter is allowed)
- Has no reasonable connection to the spec
- Writes a file but returns a DIFFERENT path than was written (e.g. uuid.uuid4() called twice)

Mark ok:true when outputs are saved with save_image / save_wav / save_bytes, moviepy
write_videofile under op_dir, or equivalent, and the same returned link strings are placed in
bundle arrays, even if the spec describes results in plain language as "images" or "texts"
rather than "paths".

Minor style issues are fine if behavior matches the spec and paths stay under base_dir.
"""

_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
# im.save(root / rel) without prior mkdir — common $code omission.
_SAVE_WITHOUT_MKDIR_RE = re.compile(
    r"^(\s*)(\w+)\.save\(\s*(root\s*/\s*\S+)\s*\)\s*$",
    re.MULTILINE,
)
_APPEND_FRESH_UUID_RE = re.compile(
    r"""\.append\(\s*f?['"][^'"]*\{uuid\.uuid4\(\)\}"""
)
_BUNDLE_ARRAY_KEYS = (
    "prompts",
    "texts",
    "images",
    "sounds",
    "videos",
    "files",
    "changes",
)
_LINK_ARRAY_KEYS = (
    "prompts",
    "texts",
    "images",
    "sounds",
    "videos",
    "files",
)
_FOR_BUNDLE_KEY_RE = re.compile(
    r"""for\s+\w+\s+in\s+bundle(?:\.get\(\s*|\[\s*)['"](\w+)['"]"""
)
_SMOKE_HARD_ERRORS = (NameError, ImportError, ModuleNotFoundError)


def _static_codegen_issues(code: str) -> str | None:
    """Fast checks before LLM validation."""
    if _APPEND_FRESH_UUID_RE.search(code):
        return (
            "Output bundle links must use the same path as the written file; "
            "do not call uuid.uuid4() again inside .append() — use one rel variable"
        )
    if "write(dest" in code and code.count("uuid.uuid4()") >= 2:
        if "append(rel)" not in code and not re.search(
            r"\.append\(\s*rel\s*\)", code
        ):
            if re.search(r'\.append\(\s*f"', code) or re.search(
                r"\.append\(\s*str\(", code
            ):
                return (
                    "Return the same rel path used for write(dest), "
                    "not a newly generated path"
                )
    return None


def _iterated_bundle_keys(code: str) -> set[str]:
    keys = set(_FOR_BUNDLE_KEY_RE.findall(code))
    return keys & set(_BUNDLE_ARRAY_KEYS)


def _empty_smoke_bundle() -> dict[str, list]:
    return {k: [] for k in _BUNDLE_ARRAY_KEYS}


def _write_smoke_text(base_dir: Path, rel: str, text: str) -> None:
    dest = base_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _write_smoke_png(base_dir: Path, rel: str) -> None:
    dest = base_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        # 1x1 RGB PNG
        dest.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc"
            b"\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(dest, format="PNG")


def _write_smoke_wav(base_dir: Path, rel: str) -> None:
    import struct
    import wave

    dest = base_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(struct.pack("<h", 0) * 100)


def _write_smoke_bytes(base_dir: Path, rel: str, data: bytes = b"smoke") -> None:
    dest = base_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _fixture_smoke_bundle(code: str, base_dir: Path, op_tag: str) -> dict[str, list]:
    bundle = _empty_smoke_bundle()
    prefix = f"{op_tag}/smoke"
    for key in _iterated_bundle_keys(code):
        if key == "texts":
            rel = f"{prefix}/sample.txt"
            _write_smoke_text(base_dir, rel, "alpha\n\nbeta\n")
            bundle["texts"] = [rel]
        elif key == "images":
            rel = f"{prefix}/sample.png"
            _write_smoke_png(base_dir, rel)
            bundle["images"] = [rel]
        elif key == "sounds":
            rel = f"{prefix}/sample.wav"
            _write_smoke_wav(base_dir, rel)
            bundle["sounds"] = [rel]
        elif key == "files":
            rel = f"{prefix}/sample.bin"
            _write_smoke_bytes(base_dir, rel)
            bundle["files"] = [rel]
        elif key == "videos":
            if "VideoFileClip" in code:
                continue
            rel = f"{prefix}/missing.mp4"
            bundle["videos"] = [rel]
        elif key == "prompts":
            rel = f"{prefix}/prompt.txt"
            _write_smoke_text(base_dir, rel, "smoke prompt")
            bundle["prompts"] = [rel]
    return bundle


def _is_smoke_hard_failure(exc: BaseException) -> bool:
    if isinstance(exc, _SMOKE_HARD_ERRORS):
        return True
    if isinstance(exc, TypeError):
        msg = str(exc).lower()
        if "run()" in msg or "required positional argument" in msg:
            return True
    return False


def _smoke_output_link_issues(result: dict, base_dir: Path) -> str | None:
    """Reject handlers that put inline text in link arrays instead of file paths."""
    root = Path(base_dir)
    for key in _LINK_ARRAY_KEYS:
        for link in result.get(key, []):
            if not isinstance(link, str):
                return f"smoke output: {key}[] values must be strings"
            if not link.strip():
                continue
            if "\n" in link or "\r" in link:
                return (
                    f"smoke output: {key}[] must hold session file links, "
                    "not inline multiline text"
                )
            path = root / link
            if path.is_file():
                continue
            if key in ("prompts", "texts") and "/" not in link and "\\" not in link:
                return (
                    f"smoke output: {key}[] must hold session file links, "
                    f"not inline text ({link[:48]!r})"
                )
    return None


def _smoke_run_once(
    code: str,
    bundle: dict[str, list],
    *,
    base_dir: Path,
    op_dir: Path,
) -> str | None:
    import inspect
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run_py = Path(tmp) / "run.py"
        run_py.write_text(code, encoding="utf-8")
        try:
            run_fn = load_run_function(run_py)
        except Exception as exc:
            return f"smoke load failed: {type(exc).__name__}: {exc}"

        params = list(inspect.signature(run_fn).parameters)
        args: list[Any] = [bundle, str(base_dir.resolve())]
        if len(params) >= 3:
            args.append(str(op_dir.resolve()))

        try:
            result = run_fn(*args)
        except Exception as exc:
            if _is_smoke_hard_failure(exc):
                return f"smoke execution failed: {type(exc).__name__}: {exc}"
            return None

    if not isinstance(result, dict):
        return (
            f"smoke execution failed: run() returned "
            f"{type(result).__name__}, expected dict"
        )
    return _smoke_output_link_issues(result, base_dir)


def smoke_execution_issues(code: str) -> str | None:
    """Run handler with empty and fixture bundles; catch import/name errors."""
    import tempfile

    if "def run" not in code:
        return "missing def run()"

    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp) / "session"
        op_dir = base_dir / "1__smoke"
        op_dir.mkdir(parents=True)

        for bundle in (
            _empty_smoke_bundle(),
            _fixture_smoke_bundle(code, base_dir, "1__smoke"),
        ):
            err = _smoke_run_once(
                code, bundle, base_dir=base_dir, op_dir=op_dir
            )
            if err:
                return err
    return None


def fixup_save_mkdir(code: str) -> tuple[str, bool]:
    """Insert dest.parent.mkdir before .save(root / …) when missing."""

    def repl(m: re.Match[str]) -> str:
        indent, obj, path_expr = m.group(1), m.group(2), m.group(3)
        return (
            f"{indent}dest = {path_expr}\n"
            f"{indent}dest.parent.mkdir(parents=True, exist_ok=True)\n"
            f"{indent}{obj}.save(dest)"
        )

    new, n = _SAVE_WITHOUT_MKDIR_RE.subn(repl, code)
    return new, n > 0


def apply_run_py_fixups(code: str) -> str:
    code, _ = fixup_save_mkdir(code)
    return code


def ensure_run_py_fixups(run_py: Path) -> None:
    text = run_py.read_text(encoding="utf-8")
    fixed = apply_run_py_fixups(text)
    if fixed != text:
        run_py.write_text(fixed.rstrip() + "\n", encoding="utf-8")
        print(
            f"custom_actions: fixed mkdir before save in {run_py}",
            file=sys.stderr,
            flush=True,
        )


def prompt_hash(spec: str) -> str:
    normalized = "\n".join(line.rstrip() for line in spec.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def custom_action_dir(repo_root: Path, name: str) -> Path:
    return repo_root / "custom_actions" / name


CUSTOM_CODE_LOG_NAME = "custome_code.txt"
CUSTOM_CODE_REJECTED_LOG_PREFIX = "custome_code_rejected_"


def rejected_attempt_log_name(attempt: int) -> str:
    return f"{CUSTOM_CODE_REJECTED_LOG_PREFIX}{attempt}.txt"


def write_rejected_attempt_log(
    op_dir: Path,
    attempt: int,
    max_attempts: int,
    code: str,
    *,
    name: str,
    spec: str,
    reason: str,
    validator_response: str = "",
) -> Path:
    """One file per failed codegen attempt; reason is first line."""
    lines = [
        f"REJECTION REASON: {reason}",
        f"attempt: {attempt}/{max_attempts}",
        f"action: &{name}",
        "",
    ]
    if spec.strip():
        lines.append("specification:")
        lines.append(spec.strip())
        lines.append("")
    if validator_response.strip():
        lines.append("validator response:")
        lines.append(validator_response.strip())
        lines.append("")
    lines.append("rejected code:")
    lines.append(code.rstrip())
    lines.append("")
    path = op_dir / rejected_attempt_log_name(attempt)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_custom_code_log(
    op_dir: Path,
    code: str,
    *,
    name: str,
    spec: str = "",
    handler_path: Path | str | None = None,
) -> Path:
    """Write accepted handler source into custome_code.txt."""
    lines = [
        f"# &{name}",
        "# validation: ok",
        "",
    ]
    if spec.strip():
        lines.append("# specification:")
        lines.extend(f"# {ln}" for ln in spec.strip().splitlines())
        lines.append("")
    if handler_path is not None:
        lines.append(f"# handler: {handler_path}")
        lines.append("")
    lines.append(code.rstrip())
    lines.append("")
    path = op_dir / CUSTOM_CODE_LOG_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _extract_python(text: str) -> str:
    text = text.strip()
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _use_custom_venv() -> bool:
    return os.environ.get("AH_EMULATE_CODE", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def _code_complete_inprocess(
    prompt: str,
    *,
    system: str,
    max_tokens: int = 4096,
    repo_root: Path | None = None,
) -> str:
    if os.environ.get("AH_EMULATE_CODE", "").lower() in ("1", "true", "yes"):
        return _emulated_run_py(prompt)

    from externals.code.model_list import get_code_llm, resolve_code_n_ctx
    from externals.llm.context_limit import estimate_tokens

    request_json = json.dumps({"task": prompt}, ensure_ascii=False)
    n_ctx = resolve_code_n_ctx(request_json, max_tokens, explicit=None)
    llm = get_code_llm("default", n_ctx=n_ctx)
    raw = llm.complete(
        request_json,
        system=system,
        max_tokens=max_tokens,
        temperature=0.1,
        seed=0,
    )
    return _extract_python(raw)


def _code_complete(
    prompt: str,
    *,
    system: str,
    max_tokens: int = 4096,
    repo_root: Path,
    op_dir: Path | None = None,
) -> str:
    if not _use_custom_venv():
        return _code_complete_inprocess(
            prompt, system=system, max_tokens=max_tokens, repo_root=repo_root
        )

    from ahlib.custom_action_env import ensure_custom_actions_env, venv_python
    from ahlib.custom_action_env import _subprocess_env as env_builder

    import subprocess
    import tempfile

    ensure_custom_actions_env(repo_root)
    py = venv_python(repo_root)
    payload = {
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        out_path = tmp_path / "out.txt"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        cmd = [
            str(py),
            "-m",
            "ahlib.custom_action_runner",
            "complete",
            "--repo-root",
            str(repo_root.resolve()),
            "--payload",
            str(payload_path),
            "--out",
            str(out_path),
        ]
        r = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env_builder(repo_root),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip()
            raise RuntimeError(
                f"custom_actions codegen failed (exit {r.returncode}): {detail}"
            )
        text = out_path.read_text(encoding="utf-8")
        if op_dir is not None:
            (op_dir / "codegen_stdout.txt").write_text(text, encoding="utf-8")
        return text


def _emulated_run_py(spec: str) -> str:
    """Minimal handler for tests without llama-cpp."""
    spec_low = spec.lower()
    crop = "100" in spec_low and ("bottom" in spec_low or "crop" in spec_low)
    if crop and "image" in spec_low:
        return _CROP_BOTTOM_TEMPLATE
    return _PASSTHROUGH_TEMPLATE


_PASSTHROUGH_TEMPLATE = '''"""Emulated custom action (pass-through)."""

def run(bundle: dict, base_dir: str, op_dir: str) -> dict:
    keys = ("prompts", "texts", "images", "sounds", "videos", "files", "changes")
    return {k: list(bundle.get(k, [])) for k in keys}
'''

_CROP_BOTTOM_TEMPLATE = '''"""Emulated custom action: crop bottom pixels from images."""

from pathlib import Path

from PIL import Image

from ahlib.custom_action_io import save_image

_CROP_PX = 100


def run(bundle: dict, base_dir: str, op_dir: str) -> dict:
    root = Path(base_dir)
    keys = ("prompts", "texts", "images", "sounds", "videos", "files", "changes")
    out = {k: list(bundle.get(k, [])) for k in keys}
    new_images: list[str] = []
    for link in bundle.get("images", []):
        im = Image.open(root / link).convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, max(1, h - _CROP_PX)))
        new_images.append(
            save_image(base_dir, op_dir, f"crop_{len(new_images)}.png", im)
        )
    out["images"] = new_images
    return out
'''


def _build_generation_prompt(
    name: str,
    spec: str,
    prior_failures: list[_FailedAttempt],
) -> str:
    parts = [
        f"Implement custom action &{name}.",
        "",
        f"SPECIFICATION:\n{spec}",
        "",
    ]
    if prior_failures:
        parts.append(
            "PREVIOUS FAILED ATTEMPTS — do not repeat these mistakes; "
            "fix every issue listed in REJECTION REASON:"
        )
        parts.append("")
        for fail in prior_failures:
            parts.extend(
                [
                    f"--- attempt {fail.attempt} ---",
                    f"REJECTION REASON: {fail.reason}",
                    "REJECTED CODE:",
                    fail.code.rstrip(),
                    "",
                ]
            )
        parts.append(
            "Write a corrected run(bundle, base_dir, op_dir) that fully satisfies "
            "the SPECIFICATION and addresses every REJECTION REASON above."
        )
    else:
        parts.append(
            "Write run(bundle, base_dir, op_dir) as documented in the system message."
        )
    return "\n".join(parts)


def _parse_validation_response(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Validator did not return JSON: {text[:500]}")
    data = json.loads(m.group(0))
    if not isinstance(data, dict) or "ok" not in data:
        raise ValueError(f"Invalid validator JSON: {data!r}")
    return data


def check_generated_code(
    code: str,
    spec: str,
    *,
    repo_root: Path | None = None,
    op_dir: Path | None = None,
) -> tuple[bool, str, str]:
    """Return (ok, reason, validator_response)."""
    static = _static_codegen_issues(code)
    if static:
        return False, static, ""

    smoke = smoke_execution_issues(code)
    if smoke:
        return False, smoke, ""

    prompt = (
        f"SPECIFICATION:\n{spec.strip()}\n\n"
        f"GENERATED PYTHON:\n{code}\n"
    )
    if os.environ.get("AH_EMULATE_CODE", "").lower() in ("1", "true", "yes"):
        forbidden = ("subprocess", "os.system", "eval(", "exec(", "__import__")
        for token in forbidden:
            if token in code:
                return False, f"forbidden {token!r}", ""
        return True, "", ""

    if repo_root is None:
        raw = _code_complete_inprocess(
            prompt, system=_VALIDATION_SYSTEM, max_tokens=512
        )
    else:
        raw = _code_complete(
            prompt,
            system=_VALIDATION_SYSTEM,
            max_tokens=512,
            repo_root=repo_root,
            op_dir=op_dir,
        )
    try:
        data = _parse_validation_response(raw)
    except ValueError as exc:
        return False, str(exc), raw
    if data.get("ok"):
        return True, "", raw
    return False, data.get("reason") or "validation failed", raw


def validate_generated_code(
    code: str,
    spec: str,
    *,
    name: str = "",
    repo_root: Path | None = None,
    op_dir: Path | None = None,
) -> None:
    ok, reason, raw = check_generated_code(
        code, spec, repo_root=repo_root, op_dir=op_dir
    )
    if ok:
        return
    if op_dir is not None:
        write_rejected_attempt_log(
            op_dir,
            1,
            1,
            code,
            name=name or "?",
            spec=spec,
            reason=reason,
            validator_response=raw,
        )
    prefix = "Emulated validation failed" if os.environ.get(
        "AH_EMULATE_CODE", ""
    ).lower() in ("1", "true", "yes") else "Custom action code rejected"
    raise ValueError(f"{prefix}: {reason}")


def run_handler_subprocess(
    run_py: Path,
    bundle: "ArrayBundle",
    session_base_dir: Path,
    op_dir: Path,
    repo_root: Path,
) -> "ArrayBundle":
    """Execute run.py inside the custom-actions venv."""
    from ahlib.ah_runtime import ArrayBundle
    from ahlib.custom_action_env import _subprocess_env as env_builder
    from ahlib.custom_action_env import ensure_imports_for_code, ensure_venv

    import subprocess

    if not _use_custom_venv():
        run_fn = load_run_function(run_py)
        import inspect

        base_dir = str(session_base_dir.resolve())
        op_dir_str = str(op_dir.resolve())
        params = list(inspect.signature(run_fn).parameters)
        if len(params) >= 3:
            raw = run_fn(bundle.as_dict(), base_dir, op_dir_str)
        else:
            raw = run_fn(bundle.as_dict(), base_dir)
        if not isinstance(raw, dict):
            raise TypeError(f"run() must return dict, got {type(raw).__name__}")
        return ArrayBundle.from_dict(raw)

    ensure_venv(repo_root)
    ensure_run_py_fixups(run_py)
    meta_path = run_py.parent / "meta.json"
    code = run_py.read_text(encoding="utf-8")
    if meta_path.is_file():
        ensure_imports_for_code(repo_root, code, meta_path)
    else:
        ensure_imports_for_code(repo_root, code, None)

    from ahlib.custom_action_env import venv_python

    py = venv_python(repo_root)
    env = env_builder(repo_root)
    env["AH_SESSION_BASE_DIR"] = str(session_base_dir.resolve())
    cmd = [
        str(py),
        "-m",
        "ahlib.custom_action_runner",
        "run",
        "--repo-root",
        str(repo_root.resolve()),
        "--run-py",
        str(run_py.resolve()),
        "--op-dir",
        str(op_dir.resolve()),
    ]
    r = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    out_path = op_dir / "output.json"
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        err_path = op_dir / "error.txt"
        if err_path.is_file():
            detail = err_path.read_text(encoding="utf-8", errors="replace") + "\n" + detail
        raise RuntimeError(
            f"&{run_py.parent.name}: handler failed (exit {r.returncode}): {detail}"
        )
    if not out_path.is_file():
        raise FileNotFoundError(f"Custom action did not write {out_path}")
    return ArrayBundle.from_dict(
        json.loads(out_path.read_text(encoding="utf-8"))
    )


def generate_and_store(
    name: str,
    spec: str,
    repo_root: Path,
    *,
    op_dir: Path | None = None,
) -> Path:
    """Create or refresh custom_actions/<name>/run.py; return path to run.py."""
    spec = spec.strip()
    if not spec:
        raise ValueError(f"&{name}: empty specification body")

    if _use_custom_venv():
        from ahlib.custom_action_env import ensure_custom_actions_env

        ensure_custom_actions_env(repo_root)

    action_dir = custom_action_dir(repo_root, name)
    action_dir.mkdir(parents=True, exist_ok=True)
    run_py = action_dir / "run.py"
    meta_path = action_dir / "meta.json"
    digest = prompt_hash(spec)

    if meta_path.is_file() and run_py.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
        if meta.get("prompt_hash") == digest and meta.get("prompt") == spec:
            print(f"&{name}: using cached handler {run_py}", file=sys.stderr, flush=True)
            ensure_run_py_fixups(run_py)
            cached_code = run_py.read_text(encoding="utf-8")
            smoke = smoke_execution_issues(cached_code)
            if smoke:
                raise ValueError(f"Custom action cached handler failed smoke test: {smoke}")
            if _use_custom_venv() and run_py.is_file():
                from ahlib.custom_action_env import sync_imports_for_code

                sync_imports_for_code(
                    repo_root, cached_code, meta_path
                )
            if op_dir is not None:
                write_custom_code_log(
                    op_dir,
                    cached_code,
                    name=name,
                    spec=spec,
                    handler_path=run_py,
                )
            return run_py

    max_attempts = _codegen_max_attempts()
    prior_failures: list[_FailedAttempt] = []
    code = ""
    for attempt in range(1, max_attempts + 1):
        gen_prompt = _build_generation_prompt(name, spec, prior_failures)
        print(
            f"&{name}: generating Python handler via $code "
            f"(attempt {attempt}/{max_attempts})",
            file=sys.stderr,
            flush=True,
        )
        code = _code_complete(
            gen_prompt,
            system=_GENERATION_SYSTEM,
            repo_root=repo_root,
            op_dir=op_dir,
        )
        if "def run" not in code:
            reason = "generated code has no def run()"
            prior_failures.append(
                _FailedAttempt(attempt=attempt, code=code, reason=reason)
            )
            if op_dir is not None:
                write_rejected_attempt_log(
                    op_dir,
                    attempt,
                    max_attempts,
                    code,
                    name=name,
                    spec=spec,
                    reason=reason,
                )
            print(f"&{name}: attempt {attempt} rejected: {reason}", file=sys.stderr)
            continue

        code = apply_run_py_fixups(code)
        print(f"&{name}: validating attempt {attempt}", file=sys.stderr, flush=True)
        ok, reason, validator_raw = check_generated_code(
            code, spec, repo_root=repo_root, op_dir=op_dir
        )
        if ok:
            break

        prior_failures.append(
            _FailedAttempt(
                attempt=attempt,
                code=code,
                reason=reason,
                validator_response=validator_raw,
            )
        )
        if op_dir is not None:
            write_rejected_attempt_log(
                op_dir,
                attempt,
                max_attempts,
                code,
                name=name,
                spec=spec,
                reason=reason,
                validator_response=validator_raw,
            )
        print(
            f"&{name}: attempt {attempt} rejected: {reason}",
            file=sys.stderr,
            flush=True,
        )
    else:
        last = prior_failures[-1].reason if prior_failures else "unknown"
        raise ValueError(
            f"Custom action code rejected after {max_attempts} attempts: {last}"
        )

    run_py.write_text(code.rstrip() + "\n", encoding="utf-8")
    if op_dir is not None:
        write_custom_code_log(
            op_dir,
            code,
            name=name,
            spec=spec,
            handler_path=run_py,
        )
    meta = {
        "name": name,
        "prompt": spec,
        "prompt_hash": digest,
        "model": os.environ.get("AH_CUSTOM_ACTIONS_CODE_MODEL", "default"),
        "venv": str(
            os.environ.get("AH_CUSTOM_ACTIONS_VENV", ".venvs/custom_actions")
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if _use_custom_venv():
        from ahlib.custom_action_env import sync_imports_for_code

        sync_imports_for_code(repo_root, code, meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if op_dir is not None:
        (op_dir / "generated_run.py").write_text(code, encoding="utf-8")
        (op_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"&{name}: saved {run_py}", file=sys.stderr, flush=True)
    return run_py


def load_run_function(run_py: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"custom_action_{run_py.parent.name}", run_py
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load custom action from {run_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run", None)
    if not callable(fn):
        raise AttributeError(f"{run_py} must define run(bundle, base_dir)")
    return fn
