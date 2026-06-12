"""Scene detection via ffmpeg frame pipe (scaled) + average hashes."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from externals.split_video.split import (
    VideoFragment,
    _reservoir_sample,
    merge_small_fragments,
)
from externals.video_audio.ffmpeg_io import _ffmpeg_cmd, _ffprobe_cmd

DEFAULT_FRAME_WIDTH = 320


def _require_deps() -> None:
    missing: list[str] = []
    try:
        import imagehash  # noqa: F401
    except ImportError:
        missing.append("imagehash")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    if missing:
        raise RuntimeError(
            "$split_video_fast requires: "
            + ", ".join(missing)
            + ".\n  tools\\setup_external_venvs.ps1\n"
            "  or: uv sync --extra split_video_fast\n"
            "  set AH_EXTERNAL_VENV_split_video_fast=.venvs/media in .env"
        )


def _parse_rate(raw: str) -> float | None:
    text = raw.strip()
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _probe_video(video_path: Path) -> tuple[float, int, int]:
    cmd = _ffprobe_cmd(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,width,height",
            "-of",
            "json",
            str(video_path),
        ]
    )
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"$split_video_fast: ffprobe failed:\n{err}") from exc

    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"no video stream: {video_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(str(stream.get("avg_frame_rate") or ""))
    if fps is None:
        fps = _parse_rate(str(stream.get("r_frame_rate") or ""))
    if fps is None:
        fps = 25.0
    return fps, width, height


def _scaled_dims(orig_w: int, orig_h: int, target_w: int) -> tuple[int, int]:
    if orig_w <= 0 or orig_h <= 0:
        return target_w, target_w
    scaled_h = max(2, (orig_h * target_w + orig_w // 2) // orig_w)
    if scaled_h % 2:
        scaled_h += 1
    return target_w, scaled_h


def _iter_scaled_frames(
    video_path: Path,
    *,
    frame_width: int,
    scaled_w: int,
    scaled_h: int,
):
    """Yield PIL RGB images decoded from ffmpeg rawvideo at ``frame_width`` px."""
    from PIL import Image

    frame_bytes = scaled_w * scaled_h * 3

    cmd = _ffmpeg_cmd(
        [
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-an",
            "-sn",
            "-vf",
            f"scale={frame_width}:-2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            yield Image.frombytes("RGB", (scaled_w, scaled_h), raw)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0 and stderr.strip():
            raise RuntimeError(f"$split_video_fast: ffmpeg failed:\n{stderr.strip()}")


def detect_fragments(
    video_path: Path,
    *,
    threshold: int,
    hash_size: int,
    min_frames: int,
    sample_count: int = 0,
    frame_width: int = DEFAULT_FRAME_WIDTH,
) -> tuple[list[VideoFragment], float]:
    """Scan scaled ffmpeg frames and return merged fragment ranges plus fps."""
    _require_deps()
    try:
        import imagehash
    except ImportError as exc:
        raise RuntimeError(
            "$split_video_fast requires Pillow and imagehash.\n"
            "  tools\\setup_external_venvs.ps1\n"
            "  or: uv sync --extra split_video_fast"
        ) from exc

    fps, orig_w, orig_h = _probe_video(video_path)
    scaled_w, scaled_h = _scaled_dims(orig_w, orig_h, frame_width)
    fragments: list[VideoFragment] = []
    current_start = 0
    first_hash: object | None = None
    prev_hash: object | None = None
    sample_pool: list[object] = []
    seen_in_fragment = 0
    frame_index = 0

    for pil in _iter_scaled_frames(
        video_path,
        frame_width=frame_width,
        scaled_w=scaled_w,
        scaled_h=scaled_h,
    ):
        image_hash = imagehash.average_hash(pil, hash_size)

        if prev_hash is not None:
            diff = abs(image_hash - prev_hash)
            if diff > threshold:
                if first_hash is not None:
                    fragments.append(
                        VideoFragment(
                            current_start,
                            frame_index,
                            first_hash,
                            prev_hash,
                            tuple(sample_pool) if sample_count > 0 else (),
                        )
                    )
                current_start = frame_index
                first_hash = image_hash
                sample_pool = [pil] if sample_count > 0 else []
                seen_in_fragment = 1 if sample_count > 0 else 0
                prev_hash = image_hash
                frame_index += 1
                continue

        if first_hash is None:
            first_hash = image_hash

        if sample_count > 0:
            seen_in_fragment += 1
            _reservoir_sample(sample_pool, pil, seen_in_fragment, sample_count)

        prev_hash = image_hash
        frame_index += 1

    if frame_index == 0:
        return [], fps

    if frame_index > current_start and first_hash is not None and prev_hash is not None:
        fragments.append(
            VideoFragment(
                current_start,
                frame_index,
                first_hash,
                prev_hash,
                tuple(sample_pool) if sample_count > 0 else (),
            )
        )

    return (
        merge_small_fragments(
            fragments, min_frames=min_frames, sample_count=sample_count
        ),
        fps,
    )
