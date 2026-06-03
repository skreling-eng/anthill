"""Build a contact-sheet JPEG preview from one video file."""

from __future__ import annotations

from pathlib import Path

from externals.video_thumbnailer.settings import ThumbnailOptions, resolve_font_path

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PreviewBuildError(RuntimeError):
    """Could not build a preview sheet for a video."""


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}B"
        size /= 1024.0
    return f"{size:.2f} YiB"


def _human_duration(seconds: float) -> str:
    total = int(float(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _human_bitrate(bits_per_second: int | float) -> str:
    return f"{int(round(float(bits_per_second) / 1000.0))} kb/s"


def _line_height(text: str, font) -> int:
    bbox = font.getbbox(text)
    return int(bbox[3] - bbox[1]) + 1


def _load_font(
    truetype_path: str | None,
    size: int,
    *,
    repo_root: Path = _REPO_ROOT,
):
    from PIL import ImageFont

    resolved = resolve_font_path(truetype_path, repo_root)
    if resolved is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(resolved), size=size)


def _track_metadata(video_path: Path) -> tuple[dict, dict, dict | None]:
    from pymediainfo import MediaInfo

    general: dict | None = None
    video: dict | None = None
    audio: dict | None = None
    for track in MediaInfo.parse(video_path).tracks:
        data = track.to_data()
        if track.track_type == "General" and general is None:
            general = data
        elif track.track_type == "Video" and video is None:
            video = data
        elif track.track_type == "Audio" and audio is None:
            audio = data
    if general is None:
        raise PreviewBuildError(f"Could not read general metadata from {video_path!s}.")
    if video is None:
        raise PreviewBuildError(f"Could not read video metadata from {video_path!s}.")
    return general, video, audio


def _video_timing(video_meta: dict) -> tuple[int, int, float, float, float, int]:
    width = int(video_meta["width"])
    height = int(video_meta["height"])
    rotation = 0
    if "rotation" in video_meta:
        rotation = int(float(video_meta["rotation"]))
    if rotation in (90, 270):
        width, height = height, width
    aspect = float(width) / float(height)
    fps = float(video_meta["frame_rate"])
    if "duration" not in video_meta:
        raise PreviewBuildError("Video metadata has no duration; cannot schedule frames.")
    duration_sec = float(video_meta["duration"]) / 1000.0
    return width, height, aspect, fps, duration_sec, rotation


def _grid_size(options: ThumbnailOptions, aspect: float) -> tuple[int, int]:
    columns = options.columns
    rows = options.rows
    if aspect < 1.0:
        if options.vertical_video_columns is not None:
            columns = options.vertical_video_columns
        if options.vertical_video_rows is not None:
            rows = options.vertical_video_rows
    return columns, rows


def _header_lines(
    video_path: Path,
    general: dict,
    video_meta: dict,
    audio_meta: dict | None,
    *,
    width: int,
    height: int,
    duration_sec: float,
    options: ThumbnailOptions,
) -> list[str]:
    file_line = f"File: {video_path.name}"
    file_size = int(general["file_size"])
    if file_size > 1024:
        size_line = (
            f"Size: {file_size} B ({_human_size(file_size)}), "
            f"Duration: {_human_duration(duration_sec)}"
        )
    else:
        size_line = f"Size: {file_size} B, Duration: {_human_duration(duration_sec)}"

    video_meta = dict(video_meta)
    video_meta["resolution"] = f"{width}x{height}"
    video_parts: list[str] = []
    for key in ("format", "resolution", "other_display_aspect_ratio", "frame_rate", "bit_rate"):
        if key not in video_meta:
            continue
        value = video_meta[key]
        if key == "other_display_aspect_ratio":
            value = f"({value[0]})"
        elif key == "frame_rate":
            value = f"{round(float(value), 2):.2f} fps"
        elif key == "bit_rate":
            value = _human_bitrate(value)
        if not video_parts:
            video_parts.append(str(value))
        elif key == "other_display_aspect_ratio":
            video_parts[-1] = f"{video_parts[-1]} {value}"
        else:
            video_parts.append(str(value))
    video_line = f"Video: {', '.join(video_parts)}"

    if audio_meta is None:
        audio_line = "Audio: None"
    else:
        audio_parts: list[str] = []
        for key in ("format", "sampling_rate", "channel_s", "bit_rate"):
            if key not in audio_meta:
                continue
            value = audio_meta[key]
            if key == "sampling_rate":
                value = f"{value} Hz"
            elif key == "channel_s":
                if int(value) == 1:
                    value = "mono"
                elif int(value) == 2:
                    value = "stereo"
                else:
                    value = f"{value} channels"
            elif key == "bit_rate":
                value = _human_bitrate(value)
            audio_parts.append(str(value))
        audio_line = f"Audio: {', '.join(audio_parts)}" if audio_parts else "Audio: None"

    lines = [file_line, size_line, video_line, audio_line]
    if options.comment_text:
        label = options.comment_label
        if not label.endswith(":"):
            label = f"{label}:"
        lines.append(f"{label} {options.comment_text}")
    return lines


def build_preview_image(video_path: Path, options: ThumbnailOptions):
    import av
    from PIL import Image, ImageColor, ImageDraw

    general, video_meta, audio_meta = _track_metadata(video_path)
    display_w, display_h, aspect, fps, duration_sec, rotation = _video_timing(video_meta)
    columns, rows = _grid_size(options, aspect)
    cell_count = columns * rows

    if options.skip_seconds >= duration_sec:
        raise PreviewBuildError(
            f"skip_seconds ({options.skip_seconds}) is not less than video duration ({duration_sec:.3f} s)."
        )
    step_sec = (duration_sec - options.skip_seconds) / cell_count
    if step_sec < 1.0 / fps:
        raise PreviewBuildError(
            f"Video is too short for {cell_count} distinct frames at {fps:.3f} fps."
        )

    bg_rgb = ImageColor.getrgb(options.background_color)
    header_rgb = ImageColor.getrgb(options.header_font_color)
    stamp_rgb = ImageColor.getrgb(options.timestamp_font_color)
    shadow_rgb = (
        ImageColor.getrgb(options.timestamp_shadow_color)
        if options.timestamp_shadow_color
        else None
    )

    header_lines: list[str] = []
    header_font = None
    header_height = 0
    line_gap = 2
    pad = options.spacing

    if not options.no_header:
        header_lines = _header_lines(
            video_path,
            general,
            video_meta,
            audio_meta,
            width=display_w,
            height=display_h,
            duration_sec=duration_sec,
            options=options,
        )
        header_font = _load_font(options.header_font, options.header_font_size)
        header_height = pad
        for line in header_lines:
            header_height += _line_height(line, header_font) + line_gap
        header_height -= line_gap

    thumb_w = int((options.width - pad * (columns + 1)) / columns)
    thumb_h = int(thumb_w / aspect)
    sheet_w = thumb_w * columns + pad * (columns + 1)
    sheet_h = header_height + thumb_h * rows + pad * (rows + 1)

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg_rgb)
    draw = ImageDraw.Draw(sheet)

    if header_lines and header_font is not None:
        y = pad
        for line in header_lines:
            draw.text((pad, y), line, fill=header_rgb, font=header_font)
            y += _line_height(line, header_font) + line_gap

    stamp_font = _load_font(options.timestamp_font, options.timestamp_font_size)
    stamp_pad_x = 2
    stamp_pad_y = 3
    shadow_dx = 1

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        capture_time = options.skip_seconds
        for row in range(rows):
            y_cell = header_height + row * thumb_h + (row + 1) * pad
            for col in range(columns):
                x_cell = col * thumb_w + (col + 1) * pad
                target_pts = int(capture_time / stream.time_base) + stream.start_time
                container.seek(target_pts, stream=stream)

                frame_image = None
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if frame.pts is not None and frame.pts >= target_pts:
                            frame_image = frame.to_image()
                            if rotation in (90, 180, 270):
                                frame_image = frame_image.rotate(-rotation, expand=True)
                            frame_image = frame_image.resize((thumb_w, thumb_h))
                            sheet.paste(frame_image, box=(x_cell, y_cell))
                            break
                    if frame_image is not None:
                        break

                label = _human_duration(capture_time)
                text_w = stamp_font.getlength(label)
                text_h = _line_height(label, stamp_font)
                tx = int(x_cell + thumb_w - text_w - stamp_pad_x)
                ty = int(y_cell + thumb_h - text_h - stamp_pad_y - shadow_dx)
                if shadow_rgb is not None:
                    draw.text(
                        (tx + shadow_dx, ty + shadow_dx),
                        label,
                        fill=shadow_rgb,
                        font=stamp_font,
                    )
                draw.text((tx, ty), label, fill=stamp_rgb, font=stamp_font)
                capture_time += step_sec
    finally:
        container.close()

    return sheet
