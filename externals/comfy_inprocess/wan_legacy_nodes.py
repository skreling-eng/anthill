"""Legacy Comfy NODE_CLASS_MAPPINGS for Wan I2V/MEGA (matches comfy_extras/nodes_wan.py)."""

from __future__ import annotations


def register_wan_legacy_nodes() -> None:
    """Register stock Wan node implementations into nodes.NODE_CLASS_MAPPINGS."""
    import nodes

    if "WanImageToVideo" not in nodes.NODE_CLASS_MAPPINGS:
        nodes.NODE_CLASS_MAPPINGS["WanImageToVideo"] = WanImageToVideoLegacy
    if "WanVaceToVideo" not in nodes.NODE_CLASS_MAPPINGS:
        nodes.NODE_CLASS_MAPPINGS["WanVaceToVideo"] = WanVaceToVideoLegacy
    if "PrimitiveInt" not in nodes.NODE_CLASS_MAPPINGS:
        nodes.NODE_CLASS_MAPPINGS["PrimitiveInt"] = PrimitiveIntLegacy


class PrimitiveIntLegacy:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0, "min": -2**63, "max": 2**63})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value: int):
        return (int(value),)


class WanImageToVideoLegacy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    FUNCTION = "execute"

    def execute(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        start_image=None,
        clip_vision_output=None,
    ):
        import torch
        import comfy.model_management
        import comfy.utils
        import node_helpers

        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image = torch.ones(
                (length, height, width, start_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype,
            ) * 0.5
            image[: start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones(
                (
                    1,
                    1,
                    latent.shape[2],
                    concat_latent_image.shape[-2],
                    concat_latent_image.shape[-1],
                ),
                device=start_image.device,
                dtype=start_image.dtype,
            )
            mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask},
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": concat_latent_image, "concat_mask": mask},
            )

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        return (positive, negative, {"samples": latent})


class WanVaceToVideoLegacy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    FUNCTION = "execute"

    def execute(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        strength,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        import torch
        import comfy.latent_formats
        import comfy.model_management
        import comfy.utils
        import node_helpers

        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video,
                    (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]),
                    value=0.5,
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat(
                [
                    reference_image,
                    comfy.latent_formats.Wan21().process_out(
                        torch.zeros_like(reference_image)
                    ),
                ],
                dim=1,
            )

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(
                mask[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]),
                    value=1.0,
                )

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(latent_length, height_mask, width_mask),
            mode="nearest-exact",
        ).squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, : reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask],
                "vace_strength": [strength],
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask],
                "vace_strength": [strength],
            },
            append=True,
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
