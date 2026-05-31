"""Tiled VAE opt-in for $image2video."""

from __future__ import annotations

import os
import unittest

from externals.comfy_inprocess.vae_tiling import (
    configure_tiled_vae_for_job,
    force_tiled_vae,
    tiled_vae_force_full_load,
)


class TestVaeTiling(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("WAN_I2V_TILED_VAE", None)
        os.environ.pop("WAN_I2V_VRAM", None)
        os.environ.pop("WAN_I2V_TILED_VAE_FULL", None)

    def test_arg_enables(self) -> None:
        configure_tiled_vae_for_job({"tiled_vae": "1"})
        self.assertTrue(force_tiled_vae())

    def test_arg_disables(self) -> None:
        os.environ["WAN_I2V_TILED_VAE"] = "1"
        configure_tiled_vae_for_job({"tiled_vae": "0"})
        self.assertFalse(force_tiled_vae())

    def test_force_full_load_opt_in(self) -> None:
        os.environ["WAN_I2V_TILED_VAE"] = "1"
        os.environ["WAN_I2V_VRAM"] = "low"
        self.assertFalse(tiled_vae_force_full_load())
        os.environ["WAN_I2V_TILED_VAE_FULL"] = "1"
        self.assertTrue(tiled_vae_force_full_load())


class TestVramConfig(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("WAN_I2V_VRAM", None)

    def test_job_lowvram(self) -> None:
        from externals.comfy_inprocess.vram_config import configure_comfy_vram_for_job

        configure_comfy_vram_for_job({"vram": "low"})
        self.assertEqual(os.environ.get("WAN_I2V_VRAM"), "low")


if __name__ == "__main__":
    unittest.main()
