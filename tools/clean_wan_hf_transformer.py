"""Remove downloaded Wan I2V transformer weight shards from the HF hub cache (keep config + aux)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

KEEP_NAMES = frozenset(
    {
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
    }
)

CACHE_DIRS = [
    Path(__file__).resolve().parents[1]
    / "models"
    / "huggingface"
    / "hub"
    / "models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers",
]


def _blob_path(link: Path, cache_root: Path) -> Path | None:
    if not link.exists() and not link.is_symlink():
        return None
    try:
        target = os.readlink(link)
    except OSError:
        return link if link.is_file() else None
    blob = (link.parent / target).resolve()
    if blob.is_file() and cache_root in blob.parents:
        return blob
    alt = cache_root / "blobs" / Path(target).name
    return alt if alt.is_file() else None


def clean_cache(cache_root: Path) -> tuple[int, int, float]:
  deleted_files = 0
  freed = 0
  snapshots = cache_root / "snapshots"
  if not snapshots.is_dir():
    return 0, 0, 0.0

  for snap in snapshots.iterdir():
    transformer = snap / "transformer"
    if not transformer.is_dir():
      continue
    for item in list(transformer.iterdir()):
      if item.name in KEEP_NAMES:
        continue
      blob = _blob_path(item, cache_root)
      size = blob.stat().st_size if blob and blob.is_file() else 0
      try:
        item.unlink(missing_ok=True)
      except OSError as exc:
        print(f"  skip link {item.name}: {exc}", file=sys.stderr)
        continue
      if blob and blob.is_file():
        try:
          blob.unlink()
          deleted_files += 1
          freed += size
          print(f"  removed blob {blob.name[:16]}… ({size / 1e9:.2f} GB)")
        except OSError as exc:
          print(f"  skip blob {blob}: {exc}", file=sys.stderr)

  blobs_dir = cache_root / "blobs"
  if blobs_dir.is_dir():
    for incomplete in blobs_dir.glob("*.incomplete"):
      try:
        size = incomplete.stat().st_size
        incomplete.unlink()
        deleted_files += 1
        freed += size
        print(f"  removed incomplete {incomplete.name[:20]}… ({size / 1e9:.2f} GB)")
      except OSError as exc:
        print(f"  skip incomplete {incomplete.name}: {exc}", file=sys.stderr)

  return deleted_files, 0, freed / 1e9


def main() -> None:
  total_files = 0
  total_gb = 0.0
  for cache in CACHE_DIRS:
    if not cache.is_dir():
      print(f"skip (not found): {cache}")
      continue
    print(f"cleaning {cache}")
    n, _, gb = clean_cache(cache)
    total_files += n
    total_gb += gb
  print(f"done: {total_files} files removed, ~{total_gb:.2f} GB freed")


if __name__ == "__main__":
  main()
