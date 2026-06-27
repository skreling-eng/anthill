"""$avatar VRAM profile must not inherit WAN_I2V_VRAM."""

from __future__ import annotations

import os
import unittest


class TestAvatarVram(unittest.TestCase):
    def tearDown(self) -> None:
        for key in ("WAN_I2V_VRAM", "AH_AVATAR_VRAM", "AH_COMFY_VRAM"):
            os.environ.pop(key, None)

    def test_avatar_defaults_normal_despite_wan_novram(self) -> None:
        os.environ["WAN_I2V_VRAM"] = "novram"
        os.environ.pop("AH_AVATAR_VRAM", None)
        os.environ.pop("AH_COMFY_VRAM", None)

        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        from externals.comfy_inprocess.vram_config import apply_avatar_vram_settings

        apply_avatar_vram_settings()
        import comfy.cli_args as cli_args

        self.assertFalse(cli_args.args.highvram)
        self.assertFalse(cli_args.args.novram)
        self.assertFalse(cli_args.args.lowvram)

    def test_apply_normal_clears_prior_novram(self) -> None:
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        import comfy.cli_args as cli_args
        import comfy.model_management as mm
        from comfy.model_management import VRAMState

        cli_args.args.novram = True
        cli_args.args.lowvram = False
        cli_args.args.highvram = False
        mm.vram_state = VRAMState.NO_VRAM
        mm.set_vram_to = VRAMState.NO_VRAM

        from externals.comfy_inprocess.vram_config import apply_avatar_vram_settings

        apply_avatar_vram_settings()
        self.assertFalse(cli_args.args.novram)
        self.assertEqual(mm.vram_state, VRAMState.NORMAL_VRAM)
        self.assertEqual(mm.set_vram_to, VRAMState.NORMAL_VRAM)

    def test_avatar_explicit_novram(self) -> None:
        os.environ["AH_AVATAR_VRAM"] = "novram"
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        from externals.comfy_inprocess.vram_config import apply_avatar_vram_settings

        apply_avatar_vram_settings()
        import comfy.cli_args as cli_args

        self.assertTrue(cli_args.args.novram)

    def test_bootstrap_avatar_ignores_wan_novram(self) -> None:
        os.environ["WAN_I2V_VRAM"] = "novram"
        os.environ.pop("AH_AVATAR_VRAM", None)
        from externals.comfy_inprocess.stubs import ensure_comfy_import_stubs

        ensure_comfy_import_stubs()
        import comfy.cli_args as cli_args

        cli_args.args.novram = False
        cli_args.args.lowvram = False
        cli_args.args.highvram = False
        import tempfile
        from pathlib import Path

        from externals.comfy_inprocess.bootstrap import bootstrap_comfy

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bootstrap_comfy(
                input_dir=tmp_path / "input",
                output_dir=tmp_path / "output",
                load_wan_wrapper=False,
                vram_profile="avatar",
            )
        self.assertFalse(cli_args.args.novram)


if __name__ == "__main__":
    unittest.main()
