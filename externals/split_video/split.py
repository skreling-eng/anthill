"""Scene-based video splitting via per-frame average hashes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_FRAMES = 5


@dataclass(frozen=True)
class VideoFragment:
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    first_hash: object
    last_hash: object
    sample_images: tuple[Any, ...] = ()

    @property
    def size(self) -> int:
        return self.end - self.start


def _reservoir_sample(pool: list[Any], item: Any, seen: int, count: int) -> None:
    """Keep a uniform random sample of size up to ``count`` while streaming items."""
    if count <= 0:
        return
    if seen <= count:
        pool.append(item)
        return
    slot = random.randint(0, seen - 1)
    if slot < count:
        pool[slot] = item


def merge_sample_images(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
    count: int,
) -> tuple[Any, ...]:
    """Shuffle combined sample images and keep up to ``count`` random frames."""
    if count <= 0:
        return ()
    pool = list(left) + list(right)
    if not pool:
        return ()
    if len(pool) <= count:
        return tuple(pool)
    return tuple(random.sample(pool, count))


def _combine(a: VideoFragment, b: VideoFragment, *, sample_count: int) -> VideoFragment:
    sample_images = (
        merge_sample_images(a.sample_images, b.sample_images, sample_count)
        if sample_count > 0
        else ()
    )
    return VideoFragment(a.start, b.end, a.first_hash, b.last_hash, sample_images)


def merge_small_fragments(
    fragments: list[VideoFragment],
    *,
    min_frames: int,
    sample_count: int = 0,
) -> list[VideoFragment]:
    """Join fragments shorter than min_frames using adjacent-small or boundary-hash rules."""
    if min_frames <= 0 or len(fragments) <= 1:
        return list(fragments)

    merged = list(fragments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            frag = merged[i]
            if frag.size >= min_frames:
                i += 1
                continue

            # 4.1 — join with next when both are short
            if i + 1 < len(merged) and merged[i + 1].size < min_frames:
                merged[i] = _combine(merged[i], merged[i + 1], sample_count=sample_count)
                del merged[i + 1]
                changed = True
                continue

            prev_diff = None
            next_diff = None
            if i > 0:
                prev_diff = abs(frag.first_hash - merged[i - 1].last_hash)
            if i + 1 < len(merged):
                next_diff = abs(frag.last_hash - merged[i + 1].first_hash)

            if prev_diff is not None and next_diff is not None:
                if prev_diff <= next_diff:
                    merged[i - 1] = _combine(
                        merged[i - 1], merged[i], sample_count=sample_count
                    )
                    del merged[i]
                else:
                    merged[i] = _combine(
                        merged[i], merged[i + 1], sample_count=sample_count
                    )
                    del merged[i + 1]
                changed = True
                continue

            if prev_diff is not None:
                merged[i - 1] = _combine(
                    merged[i - 1], merged[i], sample_count=sample_count
                )
                del merged[i]
                changed = True
                continue

            if next_diff is not None:
                merged[i] = _combine(merged[i], merged[i + 1], sample_count=sample_count)
                del merged[i + 1]
                changed = True
                continue

            i += 1
    return merged


def _require_deps() -> None:
    missing: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python (cv2)")
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
            "$split_video requires: "
            + ", ".join(missing)
            + ".\n  tools\\setup_external_venvs.ps1\n"
            "  or: uv sync --extra split_video\n"
            "  set AH_EXTERNAL_VENV_split_video=.venvs/media in .env"
        )


def detect_fragments(
    video_path: Path,
    *,
    threshold: int,
    hash_size: int,
    min_frames: int,
    sample_count: int = 0,
) -> tuple[list[VideoFragment], float]:
    """Scan video frames and return merged fragment ranges plus fps."""
    _require_deps()
    try:
        import cv2
        import imagehash
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "$split_video requires opencv-python, Pillow, and imagehash.\n"
            "  tools\\setup_external_venvs.ps1\n"
            "  or: uv sync --extra split_video"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    fragments: list[VideoFragment] = []
    current_start = 0
    first_hash: object | None = None
    prev_hash: object | None = None
    sample_pool: list[Any] = []
    seen_in_fragment = 0
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
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
    finally:
        cap.release()

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
