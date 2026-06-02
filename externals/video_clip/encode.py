"""Build a joined MP4 from input videos timed to an audio track."""

from __future__ import annotations

import shutil
from pathlib import Path

from externals.image_clip.encode import audio_duration_seconds


def max_segments(
    audio_secs: float,
    *,
    frames_per_chunk: int,
    delete_last_frames: int,
    fps: float,
) -> int:
    """How many source videos to use (matches legacy secs/((81-del)/FPS)+1)."""
    chunk_frames = frames_per_chunk - delete_last_frames
    if chunk_frames <= 0:
        raise ValueError(
            f"frames_per_chunk ({frames_per_chunk}) must exceed "
            f"delete_last_frames ({delete_last_frames})"
        )
    chunk_secs = chunk_frames / fps
    return int(audio_secs / chunk_secs) + 1


def _segment_duration(frames_per_chunk: int, delete_last_frames: int, fps: float) -> float:
    return (frames_per_chunk - delete_last_frames) / fps


def _video_segment(
    path: Path,
    *,
    fps: float,
    frames_per_chunk: int,
    delete_last_frames: int,
    flip: bool,
):
    from moviepy import VideoFileClip, vfx

    chunk_secs = _segment_duration(frames_per_chunk, delete_last_frames, fps)
    clip = VideoFileClip(str(path))
    trim_end = clip.duration
    if delete_last_frames > 0:
        trim_end -= delete_last_frames / fps
    end = min(trim_end, chunk_secs)
    if end <= 0:
        clip.close()
        return None, None
    seg = clip.subclipped(0, end).with_fps(fps)
    if flip:
        seg = seg.with_effects([vfx.MirrorX()])
    return clip, seg


def build_joined_video_mp4(
    video_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    *,
    fps: float = 25.0,
    frames_per_chunk: int = 81,
    delete_last_frames: int = 0,
    repeat: int = 120,
    alternate_flip: bool = True,
) -> Path:
    """Concatenate video segments (read directly from MP4) and mux audio."""
    from moviepy import AudioFileClip, concatenate_videoclips

    if not video_paths:
        raise ValueError("no videos")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    audio_secs = audio_duration_seconds(audio_path)
    limit = max_segments(
        audio_secs,
        frames_per_chunk=frames_per_chunk,
        delete_last_frames=delete_last_frames,
        fps=fps,
    )
    expanded = video_paths * max(1, repeat)
    selected = expanded[:limit]

    segments = []
    opened: list = []
    try:
        for index, path in enumerate(selected):
            flip = alternate_flip and (index + 1) % 2 == 1
            source, seg = _video_segment(
                path,
                fps=fps,
                frames_per_chunk=frames_per_chunk,
                delete_last_frames=delete_last_frames,
                flip=flip,
            )
            if seg is None:
                continue
            opened.append(source)
            opened.append(seg)
            segments.append(seg)

        if not segments:
            raise RuntimeError("$video_clip: no frames extracted from input videos")

        video = concatenate_videoclips(segments, method="compose")
        opened.append(video)

        if video.duration > audio_secs:
            trimmed = video.subclipped(0, audio_secs)
            opened.append(trimmed)
            video = trimmed

        audio = AudioFileClip(str(audio_path))
        opened.append(audio)
        if hasattr(video, "with_audio"):
            final = video.with_audio(audio)
        else:
            final = video.set_audio(audio)  # moviepy 1.x
        opened.append(final)

        final.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
    finally:
        seen: set[int] = set()
        for clip in reversed(opened):
            cid = id(clip)
            if cid in seen:
                continue
            seen.add(cid)
            clip.close()

    return output_path


def require_moviepy() -> None:
    try:
        import moviepy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$video_clip needs moviepy and ffmpeg.\n"
            "  uv sync --extra clip\n"
            "  or: tools\\setup_external_venvs.ps1  (uses .venvs/media with moviepy)\n"
            "Test stub: AH_EMULATE_VIDEO_CLIP=1"
        ) from exc


def require_ffmpeg() -> None:
    from externals.video_audio.ffmpeg_paths import require_ffmpeg as _require

    try:
        _require()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "$video_clip needs ffmpeg and ffprobe.\n"
            "  uv run python tools/download_ffmpeg.py\n"
            "  or install system ffmpeg on PATH"
        ) from exc
