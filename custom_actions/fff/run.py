from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ahlib.custom_action_io import apply_db_gain, float_to_int16, save_wav


def run(bundle, base_dir, op_dir):
    root = Path(base_dir)
    out = {
        k: list(bundle.get(k, []))
        for k in (
            "prompts",
            "texts",
            "images",
            "sounds",
            "videos",
            "files",
            "changes",
        )
    }
    new_sounds = []
    for link in bundle.get("sounds", []):
        sample_rate, data = wavfile.read(root / link)
        louder_data = float_to_int16(apply_db_gain(data, 10))
        new_sounds.append(
            save_wav(
                base_dir,
                op_dir,
                f"louder_{len(new_sounds)}.wav",
                sample_rate,
                louder_data,
            )
        )
    out["sounds"] = new_sounds
    return out
