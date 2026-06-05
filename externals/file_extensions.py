"""Extension → bundle array routing for $file and $folder."""

from __future__ import annotations

# Lowercase suffix (with dot) → ArrayBundle attribute name.
EXTENSION_TO_ARRAY: dict[str, str] = {
    ".mp3": "sounds",
    ".wav": "sounds",
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
    ".webp": "images",
    ".mp4": "videos",
    ".mkv": "videos",
    ".txt": "texts",
    ".ass": "texts",
}


def array_for_extension(ext: str) -> str:
    """Return the bundle array name for a file suffix (e.g. '.ass' → 'texts')."""
    return EXTENSION_TO_ARRAY.get(ext.lower(), "files")
