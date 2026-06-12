"""FAISS HNSW index storage for local video fragment search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from externals.image2embedding.embedding_format import EMBED_DIM

FAISS_SUFFIX = ".faiss"
MAP_SUFFIX = ".ahvmap.json"
INDEX_STEM = "videos"


def read_video_fps(video_path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "$create_video_index requires opencv-python.\n"
            "  uv sync --extra video_index"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else 25.0


def _require_faiss() -> None:
    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "video index requires faiss-cpu.\n"
            "  tools\\setup_external_venvs.ps1\n"
            "  or: UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media --extra video_index"
        ) from exc


def map_path_for_index(index_path: Path) -> Path:
    """Return the .ahvmap.json sibling for a .faiss index file."""
    if index_path.suffix == ".faiss":
        return index_path.with_suffix(MAP_SUFFIX)
    return index_path.with_name(index_path.name + MAP_SUFFIX)


def index_path_for_map(map_path: Path) -> Path:
    """Return the .faiss sibling for a .ahvmap.json mapping file."""
    if map_path.name.endswith(MAP_SUFFIX):
        stem = map_path.name[: -len(MAP_SUFFIX)]
        return map_path.with_name(stem + FAISS_SUFFIX)
    return map_path.with_suffix(FAISS_SUFFIX)


def build_combined_mapping(
    rows: list[tuple[Path, Path, float, int, int, list[float]]],
) -> dict:
    """Build mapping for one shared index over many videos.

    Each row: (video_path, ahvemb_path, fps, start, end, vector).
    """
    fragments: list[dict] = []
    for idx, (video_path, ahvemb_path, fps, start, end, _vec) in enumerate(rows):
        fragments.append(
            {
                "id": idx,
                "video": str(video_path.resolve()).replace("\\", "/"),
                "ahvemb": str(ahvemb_path.resolve()).replace("\\", "/"),
                "fps": fps,
                "start": start,
                "end": end,
            }
        )
    return {"dim": EMBED_DIM, "fragments": fragments}


def build_faiss_index(rows: list[tuple[Path, Path, float, int, int, list[float]]]) -> object:
    _require_faiss()
    import faiss

    if not rows:
        raise ValueError("no fragments to index")

    vectors = np.asarray([row[5] for row in rows], dtype=np.float32)
    count = len(vectors)
    index = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(64, min(count, 200))
    index.add(vectors)
    return index


def save_index_pair(
    index_path: Path,
    index: object,
    meta: dict,
) -> Path:
    _require_faiss()
    import faiss

    index_path.parent.mkdir(parents=True, exist_ok=True)
    map_path = map_path_for_index(index_path)
    faiss.write_index(index, str(index_path))
    map_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return map_path


def save_mapping_only(map_path: Path, meta: dict) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_mapping(map_path: Path) -> dict:
    meta = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"mapping must be a JSON object: {map_path}")
    return meta


def load_index_pair(index_path: Path) -> tuple[object, dict]:
    _require_faiss()
    import faiss

    map_path = map_path_for_index(index_path)
    if not map_path.is_file():
        raise FileNotFoundError(f"mapping file not found: {map_path}")
    meta = load_mapping(map_path)
    index = faiss.read_index(str(index_path))
    return index, meta


def search_index(
    index: object,
    meta: dict,
    query_vec: list[float],
    *,
    k: int,
) -> list[dict]:
    """Return search hits sorted by closeness desc."""
    fragments = meta.get("fragments", [])
    count = len(fragments)
    if count <= 0:
        return []
    k = max(1, min(k, count))
    query = np.asarray([query_vec], dtype=np.float32)
    scores, ids = index.search(query, k)
    hits: list[dict] = []
    for idx, score in zip(ids[0], scores[0]):
        if int(idx) < 0:
            continue
        frag = fragments[int(idx)]
        hits.append(
            {
                "video": str(frag["video"]),
                "ahvemb": str(frag.get("ahvemb", "")),
                "fps": float(frag.get("fps", 25.0)),
                "start": int(frag["start"]),
                "end": int(frag["end"]),
                "closeness": float(score),
            }
        )
    hits.sort(key=lambda row: row["closeness"], reverse=True)
    return hits


def brute_force_search(
    meta: dict,
    vectors: list[list[float]],
    query_vec: list[float],
    *,
    k: int,
) -> list[dict]:
    """Emulate-friendly search without faiss."""
    if not vectors:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    scores: list[tuple[int, float]] = []
    for idx, vec in enumerate(vectors):
        v = np.asarray(vec, dtype=np.float32)
        scores.append((idx, float(np.dot(q, v))))
    scores.sort(key=lambda row: row[1], reverse=True)
    k = max(1, min(k, len(scores)))
    fragments = meta["fragments"]
    hits: list[dict] = []
    for idx, closeness in scores[:k]:
        frag = fragments[idx]
        hits.append(
            {
                "video": str(frag["video"]),
                "ahvemb": str(frag.get("ahvemb", "")),
                "fps": float(frag.get("fps", 25.0)),
                "start": int(frag["start"]),
                "end": int(frag["end"]),
                "closeness": closeness,
            }
        )
    return hits
