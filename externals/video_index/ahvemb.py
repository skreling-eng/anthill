"""Parse .ahvemb sidecar files produced by $add_video_embedding_files."""

from __future__ import annotations

from pathlib import Path

from externals.image2embedding.embedding_format import unpack_siglip_embedding

AHVEMB_DIR_NAME = ".ahvemb"
AHVEMB_SUFFIX = ".ahvemb"


def ahvemb_path_for_video(video_path: Path) -> Path:
    """Return ``<video_dir>/.ahvemb/<video_stem>.ahvemb``."""
    video_path = video_path.resolve()
    return video_path.parent / AHVEMB_DIR_NAME / f"{video_path.stem}{AHVEMB_SUFFIX}"


def parse_ahvemb_file(path: Path) -> list[tuple[int, int, list[float]]]:
    """Return (start_frame, end_frame, 256-d vector) rows from an .ahvemb file."""
    rows: list[tuple[int, int, list[float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        start = int(parts[0])
        end = int(parts[1])
        vec = unpack_siglip_embedding(parts[2])
        rows.append((start, end, vec))
    return rows
