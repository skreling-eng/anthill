"""ffmpeg helpers for $split_video fragment export."""

from __future__ import annotations

from pathlib import Path

from externals.video_audio.ffmpeg_io import _run_ffmpeg


def cut_fragment(
    video_path: Path,
    output_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> Path:
    if end_frame <= start_frame:
        raise ValueError(
            f"invalid fragment range: start={start_frame}, end={end_frame}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_sec:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    _run_ffmpeg(cmd, label="$split_video")
    if not output_path.is_file():
        raise RuntimeError(f"$split_video: no output written to {output_path}")
    return output_path
