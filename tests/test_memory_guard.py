"""Wan I2V VRAM caps."""

from __future__ import annotations

import os
import unittest

from externals.comfy_inprocess.memory_guard import (
    apply_wan_memory_limits,
    cap_frames_for_vram,
    estimate_wan_latent_tokens,
)


class TestMemoryGuard(unittest.TestCase):
    def tearDown(self) -> None:
        for key in ("WAN_I2V_AUTO_CAP", "WAN_I2V_NO_FRAME_CAP", "WAN_I2V_MAX_AREA"):
            os.environ.pop(key, None)

    def test_estimate_tokens(self) -> None:
        # 49 frames @ 768x512 -> 13 latent * 96*64 spatial
        self.assertEqual(
            estimate_wan_latent_tokens(width=768, height=512, num_frames=49),
            13 * 96 * 64,
        )

    def test_cap_frames_16gb(self) -> None:
        os.environ["WAN_I2V_AUTO_CAP"] = "1"
        capped = cap_frames_for_vram(
            width=768, height=512, num_frames=49, vram_mb=16384
        )
        self.assertLessEqual(capped, 33)
        self.assertGreaterEqual(capped, 5)

    def test_cap_frames_mega_16gb(self) -> None:
        os.environ["WAN_I2V_AUTO_CAP"] = "1"
        capped = cap_frames_for_vram(
            width=768, height=512, num_frames=49, vram_mb=16384, mega=True
        )
        self.assertLessEqual(capped, 17)
        self.assertGreaterEqual(capped, 5)

    def test_no_cap_when_disabled(self) -> None:
        os.environ["WAN_I2V_NO_FRAME_CAP"] = "1"
        self.assertEqual(
            cap_frames_for_vram(width=768, height=512, num_frames=49, vram_mb=16384),
            49,
        )

    def test_apply_scales_area(self) -> None:
        os.environ["WAN_I2V_AUTO_CAP"] = "1"
        w, h, frames = apply_wan_memory_limits(
            width=768, height=512, num_frames=81
        )
        # With no CUDA in CI, only frame cap may run; with 16GB GPU both apply.
        if os.environ.get("WAN_I2V_MAX_AREA"):
            self.assertLessEqual(w * h, int(os.environ["WAN_I2V_MAX_AREA"]))
        self.assertLessEqual(frames, 81)


if __name__ == "__main__":
    unittest.main()
