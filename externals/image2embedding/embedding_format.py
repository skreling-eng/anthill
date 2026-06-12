"""SigLIP embedding pack: 768-d float32 -> 256-d float16 -> base64 text."""

from __future__ import annotations

import base64
import hashlib

import numpy as np

SOURCE_DIM = 768
EMBED_DIM = 256

_PROJ: dict[int, np.ndarray] = {}


def _projection(source_dim: int = SOURCE_DIM) -> np.ndarray:
    """Fixed orthonormal source_dim x 256 matrix for deterministic reduction."""
    proj = _PROJ.get(source_dim)
    if proj is None:
        seed = 0 if source_dim == SOURCE_DIM else source_dim
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((source_dim, EMBED_DIM), dtype=np.float32)
        q, _ = np.linalg.qr(mat)
        proj = q[:, :EMBED_DIM]
        _PROJ[source_dim] = proj
    return proj


def reduce_siglip_vector(vec, *, source_dim: int = SOURCE_DIM) -> np.ndarray:
    """L2-normalized SigLIP vector -> L2-normalized 256-d float32."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != source_dim:
        raise ValueError(f"expected {source_dim}-d SigLIP vector, got {arr.shape[0]}")
    reduced = arr @ _projection(source_dim)
    norm = float(np.linalg.norm(reduced))
    if norm:
        reduced /= norm
    return reduced


def pack_siglip_embedding(vec, *, source_dim: int = SOURCE_DIM) -> str:
    """Encode a SigLIP vector as base64 little-endian float16 x256."""
    packed = reduce_siglip_vector(vec, source_dim=source_dim).astype(np.float16)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def pack_averaged_siglip_embeddings(vec_list, *, source_dim: int = SOURCE_DIM) -> str:
    """Mean of SigLIP vectors, L2-normalize, then pack as float16 base64."""
    if not vec_list:
        raise ValueError("no frame embeddings to average")
    stack = np.stack(
        [np.asarray(v, dtype=np.float32).reshape(-1) for v in vec_list],
        axis=0,
    )
    mean = stack.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm:
        mean /= norm
    return pack_siglip_embedding(mean, source_dim=source_dim)


def unpack_siglip_embedding(encoded: str) -> list[float]:
    """Decode base64 float16 x256 back to Python floats (promoted to float32)."""
    raw = base64.b64decode(encoded.strip(), validate=True)
    expected = EMBED_DIM * np.dtype(np.float16).itemsize
    if len(raw) != expected:
        raise ValueError(
            f"expected {expected} bytes ({EMBED_DIM} float16 values), got {len(raw)}"
        )
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.astype(np.float32).tolist()


def emulated_siglip_embedding(seed: str) -> str:
    """Deterministic placeholder embedding for emulate mode."""
    digest = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    vec = rng.standard_normal(SOURCE_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return pack_siglip_embedding(vec)
