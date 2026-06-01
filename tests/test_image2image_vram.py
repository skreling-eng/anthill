"""$image2image VRAM profile must not inherit WAN_I2V_VRAM."""

from __future__ import annotations

import os
import unittest


class TestImage2ImageVram(unittest.TestCase):
    def tearDown(self) -> None:
        for key in ("WAN_I2V_VRAM", "AH_IMAGE2IMAGE_VRAM", "AH_COMFY_VRAM"):
            os.environ.pop(key, None)

    def test_image2image_defaults_normal_despite_wan_novram(self) -> None:
        os.environ["WAN_I2V_VRAM"] = "novram"
        os.environ.pop("AH_IMAGE2IMAGE_VRAM", None)
        os.environ.pop("AH_COMFY_VRAM", None)

        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        from externals.comfy_inprocess.vram_config import apply_image2image_vram_settings

        apply_image2image_vram_settings()
        import comfy.cli_args as cli_args

        self.assertFalse(cli_args.args.highvram)
        self.assertFalse(cli_args.args.novram)
        self.assertFalse(cli_args.args.lowvram)

    def test_image2image_explicit_highvram(self) -> None:
        os.environ["AH_IMAGE2IMAGE_VRAM"] = "high"
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        from externals.comfy_inprocess.vram_config import apply_image2image_vram_settings

        apply_image2image_vram_settings()
        import comfy.cli_args as cli_args

        self.assertTrue(cli_args.args.highvram)


if __name__ == "__main__":
    unittest.main()
