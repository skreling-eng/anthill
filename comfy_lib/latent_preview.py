"""Latent preview for KSampler (headless Anthill: progress only, no UI previews)."""

from __future__ import annotations

from typing import Any


def set_preview_method(*_args, **_kwargs) -> None:
    pass


class LatentPreviewer:
    """Stub base class so VideoHelperSuite / Wrapper can import; no previews emitted."""

    def decode_latent_to_preview_image(self, preview_format, x0):
        return None


def get_previewer(device, latent_format, *args, **kwargs):
    """Headless: no TAESD / RGB latent previews (saves VRAM and CPU)."""
    return None


def prepare_callback(model, steps: int, x0_output_dict: dict[str, Any] | None = None):
    import comfy.utils

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        pbar.update_absolute(step + 1, total_steps, None)

    return callback
