import random
from pathlib import Path

def run(bundle: dict, base_dir: str, op_dir: str) -> dict:
    root = Path(base_dir)
    out = {k: list(bundle.get(k, [])) for k in
           ("prompts", "texts", "images", "sounds", "videos", "files", "changes")}
    
    # Shuffle elements in each context array
    random.shuffle(out["prompts"])
    random.shuffle(out["texts"])
    random.shuffle(out["images"])
    random.shuffle(out["sounds"])
    random.shuffle(out["videos"])
    random.shuffle(out["files"])
    
    return out
