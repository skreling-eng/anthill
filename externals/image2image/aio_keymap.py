"""Comfy AIO → diffusers state dict key remapping for Qwen-Image-Edit."""

from __future__ import annotations

from typing import Mapping

import torch


def remap_text_encoder_key(name: str) -> str | None:
    """Map Comfy qwen25_7b.transformer.* keys to Qwen2_5_VLForConditionalGeneration."""
    if name == "logit_scale":
        return None
    if name.startswith("visual."):
        return f"model.{name}"
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    return name


def remap_text_encoder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        mapped = remap_text_encoder_key(key)
        if mapped is not None:
            out[mapped] = tensor
    embed = out.get("model.language_model.embed_tokens.weight")
    if embed is not None:
        out["lm_head.weight"] = embed
    return out


def convert_comfy_vae_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Comfy Qwen/Wan VAE → diffusers AutoencoderKLQwenImage (Wan2GP converter)."""
    middle_key_mapping = {
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }
    attention_mapping = {
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }
    head_mapping = {
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }
    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in middle_key_mapping:
            out[middle_key_mapping[key]] = value
            continue
        if key in attention_mapping:
            out[attention_mapping[key]] = value
            continue
        if key in head_mapping:
            out[head_mapping[key]] = value
            continue
        if key in quant_mapping:
            out[quant_mapping[key]] = value
            continue
        if key == "encoder.conv1.weight":
            out["encoder.conv_in.weight"] = value
            continue
        if key == "encoder.conv1.bias":
            out["encoder.conv_in.bias"] = value
            continue
        if key == "decoder.conv1.weight":
            out["decoder.conv_in.weight"] = value
            continue
        if key == "decoder.conv1.bias":
            out["decoder.conv_in.bias"] = value
            continue

        if key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
            out[new_key] = value
            continue

        if key.startswith("decoder.upsamples."):
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                block_idx = int(parts[2])
                if "residual" in key:
                    if block_idx in (0, 1, 2):
                        up_block_id, resnet_id = 0, block_idx
                    elif block_idx in (4, 5, 6):
                        up_block_id, resnet_id = 1, block_idx - 4
                    elif block_idx in (8, 9, 10):
                        up_block_id, resnet_id = 2, block_idx - 8
                    elif block_idx in (12, 13, 14):
                        up_block_id, resnet_id = 3, block_idx - 12
                    else:
                        out[key] = value
                        continue
                    if ".residual.0.gamma" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.norm1.gamma"
                    elif ".residual.2.bias" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv1.bias"
                    elif ".residual.2.weight" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv1.weight"
                    elif ".residual.3.gamma" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.norm2.gamma"
                    elif ".residual.6.bias" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv2.bias"
                    elif ".residual.6.weight" in key:
                        new_key = f"decoder.up_blocks.{up_block_id}.resnets.{resnet_id}.conv2.weight"
                    else:
                        new_key = key
                    out[new_key] = value
                    continue
                if ".shortcut." in key:
                    if block_idx == 4:
                        new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                        new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                    else:
                        new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                        new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
                    out[new_key] = value
                    continue
                if (".resample." in key) or (".time_conv." in key):
                    if block_idx == 3:
                        new_key = key.replace("decoder.upsamples.3", "decoder.up_blocks.0.upsamplers.0")
                    elif block_idx == 7:
                        new_key = key.replace("decoder.upsamples.7", "decoder.up_blocks.1.upsamplers.0")
                    elif block_idx == 11:
                        new_key = key.replace("decoder.upsamples.11", "decoder.up_blocks.2.upsamplers.0")
                    else:
                        new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    out[new_key] = value
                    continue
            out[key.replace("decoder.upsamples.", "decoder.up_blocks.")] = value
            continue

        out[key] = value
    return out
