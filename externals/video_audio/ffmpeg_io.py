"""ffmpeg helpers for $detach_audio, $attach_audio, and $add_soft_subtitles."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from externals.video_audio.ffmpeg_paths import get_ffmpeg_exe, get_ffprobe_exe, require_ffmpeg

__all__ = ["require_ffmpeg", "get_ffmpeg_exe"]


SUBTITLE_FILE_SUFFIXES = frozenset({".ass", ".ssa", ".srt"})


def pair_videos_and_text_links(
    videos: list[Path], text_links: list[str]
) -> list[tuple[Path, str]]:
    """Pair each video with a texts[] link string (same rules as pair_videos_and_texts)."""
    if not videos:
        return []
    if not text_links:
        return [(video, "") for video in videos]
    if len(text_links) == 1:
        return [(video, text_links[0]) for video in videos]
    if len(videos) == 1:
        return [(videos[0], link) for link in text_links]
    if len(videos) == len(text_links):
        return list(zip(videos, text_links, strict=True))
    if len(text_links) < len(videos):
        padded = list(text_links)
        padded.extend([text_links[-1]] * (len(videos) - len(text_links)))
        return list(zip(videos, padded, strict=True))
    if len(text_links) > len(videos):
        return list(zip(videos, text_links[: len(videos)], strict=True))
    return []


def pair_videos_and_texts(
    videos: list[Path], texts: list[str]
) -> list[tuple[Path, str]]:
    """Pair each video with subtitle text (zip, broadcast, or pad shorter texts[])."""
    if not videos:
        return []
    if not texts:
        return [(video, "") for video in videos]
    if len(texts) == 1:
        return [(video, texts[0]) for video in videos]
    if len(videos) == 1:
        return [(videos[0], text) for text in texts]
    if len(videos) == len(texts):
        return list(zip(videos, texts, strict=True))
    if len(texts) < len(videos):
        padded = list(texts)
        padded.extend([texts[-1]] * (len(videos) - len(texts)))
        return list(zip(videos, padded, strict=True))
    if len(texts) > len(videos):
        return list(zip(videos, texts[: len(videos)], strict=True))
    return []


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


def _ffprobe_cmd(argv: list[str]) -> list[str]:
    """Build argv with vendored or PATH ffprobe as argv[0]."""
    return [get_ffprobe_exe(), *argv]


def _run_ffmpeg(cmd: list[str], *, label: str, cwd: Path | None = None) -> None:
    if cmd and cmd[0] == "ffmpeg":
        cmd = _ffmpeg_cmd(cmd[1:])
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        if err:
            lines = [ln for ln in err.splitlines() if ln.strip()]
            hint = "\n".join(lines[-6:])
        else:
            hint = str(exc)
        raise RuntimeError(f"{label} failed:\n{hint}") from exc


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


def _relative_posix_path(path: Path, base: Path) -> str:
    """Path relative to base using forward slashes (safe in ffmpeg -vf on Windows)."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def _strip_subtitle_text(text: str) -> str:
    return text.lstrip("\ufeff \t\r\n")


def is_subtitle_file(path: Path) -> bool:
    return path.suffix.lower() in SUBTITLE_FILE_SUFFIXES


def is_ass_subtitle(text: str) -> bool:
    """True when text looks like Advanced SubStation Alpha (.ass) content."""
    s = _strip_subtitle_text(text)
    if not s:
        return False
    if s.startswith("[Script Info]"):
        return True
    if s.startswith("Dialogue:") or "\nDialogue:" in s[:16000]:
        return True
    return "[Events]" in s[:4000] and "Format:" in s[:4000]


def _stage_font(work_dir: Path, font_path: Path | None) -> str | None:
    """Copy font into work_dir so drawtext can use a relative fontfile= path."""
    if font_path is None:
        return None
    fonts_dir = work_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    dest = fonts_dir / "subtitle_font.ttf"
    if not dest.exists() or dest.stat().st_mtime < font_path.stat().st_mtime:
        dest.write_bytes(font_path.read_bytes())
    return _relative_posix_path(dest, work_dir)


def add_soft_subtitles(
    video_path: Path,
    output_path: Path,
    text: str,
    *,
    work_dir: Path,
    subtitle_file: Path,
    font_path: Path | None,
    subtitle_src: Path | None = None,
    font_size: int = 40,
    bottom_margin: int = 40,
) -> Path:
    """Burn subtitles onto a video (ASS/SRT via subtitles=, plain text via drawtext=)."""
    work_dir = work_dir.resolve()
    output_path = output_path.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subtitle_file.parent.mkdir(parents=True, exist_ok=True)
    rel_out = _relative_posix_path(output_path, work_dir)

    if subtitle_src is not None and is_subtitle_file(subtitle_src):
        dest = subtitle_file.with_suffix(subtitle_src.suffix.lower())
        shutil.copy2(subtitle_src, dest)
        rel_sub = _relative_posix_path(dest, work_dir)
        vf = f"subtitles={rel_sub}"
    elif is_ass_subtitle(text):
        dest = subtitle_file.with_suffix(".ass")
        dest.write_text(_strip_subtitle_text(text), encoding="utf-8-sig")
        rel_sub = _relative_posix_path(dest, work_dir)
        vf = f"subtitles={rel_sub}"
    else:
        dest = subtitle_file.with_suffix(".txt")
        dest.write_text(_prepare_drawtext_content(text), encoding="utf-8")
        rel_sub = _relative_posix_path(dest, work_dir)
        font_rel = _stage_font(work_dir, font_path)
        if font_rel is None:
            raise RuntimeError(
                "$add_soft_subtitles: drawtext needs font= pointing to a .ttf "
                "(or pass .ass/.srt subtitle files in texts[])"
            )
        parts: list[str] = [
            f"fontfile={font_rel}",
            f"textfile={rel_sub}",
            f"fontsize={font_size}",
            "fontcolor=white",
            "borderw=2",
            "bordercolor=black@0.8",
            "box=1",
            "boxcolor=black@0.5",
            "boxborderw=8",
            "x=(w-text_w)/2",
            f"y=h-th-{bottom_margin}",
        ]
        vf = "drawtext=" + ":".join(parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path.resolve()),
        "-vf",
        vf,
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
        rel_out,
    ]
    _run_ffmpeg(cmd, label="$add_soft_subtitles", cwd=work_dir)
    if not output_path.is_file():
        raise RuntimeError(f"$add_soft_subtitles: no output written to {output_path}")
    return output_path


def _prepare_drawtext_content(text: str) -> str:
    import re

    text = _strip_subtitle_text(text)
    if len(text) > 10:
        text = re.sub(r"\^", "\n", text)
    return text
