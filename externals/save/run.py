"""$save — write array data to a file path."""

from __future__ import annotations

from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    path = inp.args.get("_path", inp.args.get("path", "output.bin"))
    ext = Path(path).suffix.lower()
    source_key = {
        ".mp4": "videos",
        ".png": "images",
        ".jpg": "images",
        ".jpeg": "images",
        ".mp3": "sounds",
        ".txt": "texts",
    }.get(ext, "files")
    src = getattr(inp.bundle, source_key)
    save_dir = ctx.op_dir / "saved"
    save_dir.mkdir(exist_ok=True)
    dest = save_dir / Path(path).name
    if src:
        data = ctx.read_link_bytes(src[-1])
        dest.write_bytes(data)
    else:
        dest.write_text(f"[emulated empty $save to {path}]\n", encoding="utf-8")
    link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
    out.files.append(link)
    return out
