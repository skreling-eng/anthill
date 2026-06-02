"""Build a slideshow MP4 from images timed to an audio track."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image


def audio_duration_seconds(path: Path) -> float:
    """Duration via ffprobe (handles float WAV, mp3, etc.)."""
    from externals.video_audio.ffmpeg_paths import get_ffprobe_exe

    proc = subprocess.run(
        [
            get_ffprobe_exe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def _frame_counts(duration: float, image_count: int, fps: float) -> list[int]:
    """Split total frames across images so video length matches audio at fps."""
    total = max(image_count, round(duration * fps))
    base, extra = divmod(total, image_count)
    return [base + (1 if i < extra else 0) for i in range(image_count)]


def build_slideshow_mp4(
    image_paths: list[Path],
    audio_path: Path,
    *,
    fps: float = 25.0,
    work_dir: Path,
) -> Path:
    if not image_paths:
        raise ValueError("no images")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    duration = audio_duration_seconds(audio_path)
    counts = _frame_counts(duration, len(image_paths), fps)

    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(image_paths[0]).convert("RGB")
    size = first.size
    frame_idx = 0
    for img_path, n_frames in zip(image_paths, counts):
        img = Image.open(img_path).convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        for _ in range(n_frames):
            out = frames_dir / f"frame_{frame_idx:06d}.png"
            img.save(out)
            frame_idx += 1

    from externals.video_audio.ffmpeg_paths import get_ffmpeg_exe

    silent = work_dir / "silent.mp4"
    subprocess.run(
        [
            get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(silent),
        ],
        check=True,
        capture_output=True,
    )

    output = work_dir / "clip.mp4"
    mux_audio_with_moviepy(silent, audio_path, output, fps=fps)
    return output


def mux_audio_with_moviepy(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    fps: float,
) -> None:
    from moviepy import AudioFileClip, VideoFileClip

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))
    final = None
    try:
        if hasattr(video, "with_audio"):
            final = video.with_audio(audio)
        else:
            final = video.set_audio(audio)  # moviepy 1.x
        final.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
    finally:
        for clip in (final, audio, video):
            if clip is not None:
                clip.close()


def require_moviepy() -> None:
    try:
        import moviepy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$image_clip needs moviepy and ffmpeg.\n"
            "  uv sync --extra clip\n"
            "  or: tools\\setup_external_venvs.ps1  (uses .venvs/media with moviepy)\n"
            "Test stub: AH_EMULATE_IMAGE_CLIP=1"
        ) from exc


def require_ffmpeg() -> None:
    from externals.video_audio.ffmpeg_paths import require_ffmpeg as _require

    try:
        _require()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "$image_clip needs ffmpeg and ffprobe.\n"
            "  uv run python tools/download_ffmpeg.py\n"
            "  or install system ffmpeg on PATH"
        ) from exc
