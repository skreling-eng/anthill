"""Sample frames from video files for $video2embedding."""

from __future__ import annotations

from pathlib import Path

DEFAULT_TARGET_FRAMES = 5


def every_nth_for_frame_count(frame_count: int, target: int = DEFAULT_TARGET_FRAMES) -> int:
    """Return every=N so sampling yields about ``target`` frames from ``frame_count`` frames."""
    if frame_count <= 0:
        return 1
    if frame_count <= target:
        return 1
    return max(1, frame_count // target)


def sample_video_frames(
    video_path: Path,
    *,
    every_nth: int | None = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> list[object]:
    """Return PIL RGB images for every Nth frame (1-based: N, 2N, 3N, …)."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "$video2embedding requires opencv-python.\n"
            "  uv sync --extra video2embedding\n"
            "  or: uv sync --extra media"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        if every_nth is None:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            every_nth = (
                every_nth_for_frame_count(total, target_frames)
                if total > 0
                else 1
            )
    finally:
        cap.release()

    if every_nth < 1:
        raise ValueError(f"every must be >= 1, got {every_nth}")

    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frames: list[object] = []
    first_frame = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if first_frame is None:
                first_frame = pil
            if index % every_nth == 0:
                frames.append(pil)
    finally:
        cap.release()

    if not frames and first_frame is not None:
        frames.append(first_frame)
    return frames


def sample_fragment_frames(
    video_path: Path,
    start: int,
    end: int,
    *,
    every_nth: int | None = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> list[object]:
    """Return PIL RGB images for every Nth frame within [start, end) (0-based, end exclusive)."""
    if every_nth is None:
        every_nth = every_nth_for_frame_count(end - start, target_frames)
    if every_nth < 1:
        raise ValueError(f"every must be >= 1, got {every_nth}")
    if end <= start:
        return []

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "$video2embedding requires opencv-python.\n"
            "  uv sync --extra video2embedding\n"
            "  or: uv sync --extra media"
        ) from exc

    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frames: list[object] = []
    first_in_fragment = None
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame_index >= end:
                break
            if start <= frame_index < end:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                if first_in_fragment is None:
                    first_in_fragment = pil
                if (frame_index + 1) % every_nth == 0:
                    frames.append(pil)
            frame_index += 1
    finally:
        cap.release()

    if not frames and first_in_fragment is not None:
        frames.append(first_in_fragment)
    return frames


def read_video_frames(
    video_path: Path,
    frame_indices: list[int] | tuple[int, ...],
) -> list[object]:
    """Return PIL RGB images for specific 0-based frame indices (in request order)."""
    if not frame_indices:
        return []

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "$video2embedding requires opencv-python.\n"
            "  uv sync --extra video2embedding\n"
            "  or: uv sync --extra media"
        ) from exc

    from PIL import Image

    wanted = list(frame_indices)
    wanted_set = set(wanted)
    max_idx = max(wanted_set)
    by_index: dict[int, object] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frame_index = 0
    try:
        while frame_index <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in wanted_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                by_index[frame_index] = Image.fromarray(rgb)
            frame_index += 1
    finally:
        cap.release()

    return [by_index[i] for i in wanted if i in by_index]
