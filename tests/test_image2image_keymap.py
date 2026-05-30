"""AIO → diffusers key remapping tests."""

from __future__ import annotations

import unittest

import torch

from externals.image2image.aio_keymap import (
    convert_comfy_vae_state_dict,
    remap_text_encoder_key,
    remap_text_encoder_state,
)


class TestTextEncoderKeymap(unittest.TestCase):
    def test_visual_prefix(self) -> None:
        self.assertEqual(
            remap_text_encoder_key("visual.patch_embed.proj.weight"),
            "model.visual.patch_embed.proj.weight",
        )

    def test_language_model_prefix(self) -> None:
        self.assertEqual(
            remap_text_encoder_key("model.layers.0.self_attn.q_proj.weight"),
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )

    def test_logit_scale_skipped(self) -> None:
        self.assertIsNone(remap_text_encoder_key("logit_scale"))

    def test_lm_head_tied_to_embed(self) -> None:
        embed = torch.zeros(2, 3)
        state = remap_text_encoder_state(
            {
                "model.embed_tokens.weight": embed,
                "visual.blocks.0.weight": torch.zeros(1),
            }
        )
        self.assertIn("model.language_model.embed_tokens.weight", state)
        self.assertIn("model.visual.blocks.0.weight", state)
        self.assertTrue(torch.equal(state["lm_head.weight"], embed))
        self.assertNotIn("logit_scale", state)


class TestVaeKeymap(unittest.TestCase):
    def test_quant_and_conv_in(self) -> None:
        w = torch.zeros(1)
        out = convert_comfy_vae_state_dict(
            {
                "conv1.weight": w,
                "encoder.conv1.bias": w,
                "decoder.conv1.weight": w,
            }
        )
        self.assertIn("quant_conv.weight", out)
        self.assertIn("encoder.conv_in.bias", out)
        self.assertIn("decoder.conv_in.weight", out)

    def test_encoder_downsample_residual(self) -> None:
        w = torch.zeros(1)
        out = convert_comfy_vae_state_dict(
            {"encoder.downsamples.0.residual.2.weight": w}
        )
        self.assertIn("encoder.down_blocks.0.conv1.weight", out)

    def test_decoder_upsample_residual_block_map(self) -> None:
        w = torch.zeros(1)
        out = convert_comfy_vae_state_dict(
            {"decoder.upsamples.5.residual.6.weight": w}
        )
        self.assertIn("decoder.up_blocks.1.resnets.1.conv2.weight", out)


if __name__ == "__main__":
    unittest.main()
