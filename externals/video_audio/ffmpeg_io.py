"""ffmpeg helpers for $detach_audio and $attach_audio."""

from __future__ import annotations

import subprocess
from pathlib import Path

from externals.video_audio.ffmpeg_paths import get_ffmpeg_exe, require_ffmpeg

__all__ = ["require_ffmpeg", "get_ffmpeg_exe"]


def pair_videos_and_sounds(
    videos: list[Path], sounds: list[Path]
) -> list[tuple[Path, Path]]:
    """Pair clips for mux: equal length, or broadcast when one side has length 1."""
    if not videos or not sounds:
        return []
    if len(videos) == len(sounds):
        return list(zip(videos, sounds, strict=True))
    if len(videos) == 1:
        return [(videos[0], sound) for sound in sounds]
    if len(sounds) == 1:
        return [(video, sounds[0]) for video in videos]
    raise ValueError(
        f"Need equal videos[] and sounds[], or one side length 1; "
        f"got {len(videos)} video(s) and {len(sounds)} sound(s)"
    )


def _ffmpeg_cmd(argv: list[str]) -> list[str]:
    """Build argv with vendored or PATH ffmpeg as argv[0]."""
    return [get_ffmpeg_exe(), *argv]


def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    if cmd and cmd[0] == "ffmpeg":
        cmd = _ffmpeg_cmd(cmd[1:])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        hint = err.splitlines()[-1] if err else str(exc)
        raise RuntimeError(f"{label} failed: {hint}") from exc


def detach_audio(
    video_path: Path,
    output_path: Path,
    *,
    fmt: str = "wav",
) -> Path:
    """Extract the audio track from a video file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key = fmt.strip().lower()
    if key in ("wav", "wave"):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
    elif key == "copy":
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "copy",
            str(output_path),
        ]
    elif key == "aac":
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "aac",
            str(output_path),
        ]
    elif key == "mp3":
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ]
    else:
        raise ValueError(f"unsupported audio format {fmt!r} (use wav, aac, mp3, or copy)")

    _run_ffmpeg(cmd, label="$detach_audio")
    if not output_path.is_file():
        raise RuntimeError(f"$detach_audio: no output written to {output_path}")
    return output_path


def attach_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    shortest: bool = True,
    audio_codec: str = "aac",
) -> Path:
    """Mux an audio file onto a video (video stream copied, audio replaced)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(str(output_path))
    _run_ffmpeg(cmd, label="$attach_audio")
    if not output_path.is_file():
        raise RuntimeError(f"$attach_audio: no output written to {output_path}")
    return output_path
