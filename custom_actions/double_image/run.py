"""Emulated custom action (pass-through)."""

def run(bundle: dict, base_dir: str, op_dir: str) -> dict:
    keys = ("prompts", "texts", "images", "sounds", "videos", "files", "changes")
    return {k: list(bundle.get(k, [])) for k in keys}
