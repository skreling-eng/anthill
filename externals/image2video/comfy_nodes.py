"""Comfy node handlers for Wan MEGA I2V (no comfy_api dependency)."""

from __future__ import annotations

from typing import Any


def primitive_int(inputs: dict[str, Any]) -> tuple:
    return (int(inputs["value"]),)


def wan_vace_start_to_end_frame(inputs: dict[str, Any]) -> tuple:
    """WanVideoVACEStartToEndFrame — build control video + mask from start/end images."""
    import torch
    import comfy.utils

    num_frames = int(inputs["num_frames"])
    empty_frame_level = float(inputs.get("empty_frame_level", 0.5))
    start_image = inputs.get("start_image")
    end_image = inputs.get("end_image")
    control_images = inputs.get("control_images")
    inpaint_mask = inputs.get("inpaint_mask")
    start_index = int(inputs.get("start_index", 0))
    end_index = int(inputs.get("end_index", -1))

    if start_image is None and end_image is None and control_images is not None:
        if control_images.shape[0] >= num_frames:
            control_images = control_images[:num_frames]
        elif control_images.shape[0] < num_frames:
            padding = torch.ones(
                (
                    num_frames - control_images.shape[0],
                    control_images.shape[1],
                    control_images.shape[2],
                    control_images.shape[3],
                ),
                device=control_images.device,
            ) * empty_frame_level
            control_images = torch.cat([control_images, padding], dim=0)
        return (control_images.cpu().float(), torch.zeros_like(control_images[:, :, :, 0]).cpu().float())

    _shape = start_image.shape if start_image is not None else end_image.shape
    _device = start_image.device if start_image is not None else end_image.device
    height, width = int(_shape[1]), int(_shape[2])

    if end_index < 0:
        end_index = num_frames + end_index

    out_batch = torch.ones((num_frames, height, width, 3), device=_device) * empty_frame_level
    masks = torch.ones((num_frames, height, width), device=_device)

    if end_image is not None and (end_image.shape[1] != height or end_image.shape[2] != width):
        end_image = comfy.utils.common_upscale(
            end_image.movedim(-1, 1), width, height, "lanczos", "disabled"
        ).movedim(1, -1)

    if control_images is not None and (
        control_images.shape[1] != height or control_images.shape[2] != width
    ):
        control_images = comfy.utils.common_upscale(
            control_images.movedim(-1, 1), width, height, "lanczos", "disabled"
        ).movedim(1, -1)

    if start_image is not None:
        frames_to_copy = min(start_image.shape[0], num_frames - start_index)
        if frames_to_copy > 0:
            out_batch[start_index : start_index + frames_to_copy] = start_image[:frames_to_copy]
            masks[start_index : start_index + frames_to_copy] = 0

    if end_image is not None:
        end_start = end_index - end_image.shape[0] + 1
        if end_start < 0:
            end_image = end_image[abs(end_start) :]
            end_start = 0
        frames_to_copy = min(end_image.shape[0], num_frames - end_start)
        if frames_to_copy > 0:
            out_batch[end_start : end_start + frames_to_copy] = end_image[:frames_to_copy]
            masks[end_start : end_start + frames_to_copy] = 0

    if control_images is not None:
        empty_frames = masks.sum(dim=(1, 2)) > 0.5 * height * width
        if empty_frames.any():
            control_length = control_images.shape[0]
            for frame_idx in range(num_frames):
                if empty_frames[frame_idx] and frame_idx < control_length:
                    out_batch[frame_idx] = control_images[frame_idx]

    if inpaint_mask is not None:
        inpaint_mask = comfy.utils.common_upscale(
            inpaint_mask.unsqueeze(1), width, height, "nearest-exact", "disabled"
        ).squeeze(1).to(_device)
        if inpaint_mask.shape[0] > num_frames:
            inpaint_mask = inpaint_mask[:num_frames]
        elif inpaint_mask.shape[0] < num_frames:
            repeat_factor = (num_frames + inpaint_mask.shape[0] - 1) // inpaint_mask.shape[0]
            inpaint_mask = inpaint_mask.repeat(repeat_factor, 1, 1)[:num_frames]
        masks = inpaint_mask * masks

    return (out_batch.cpu().float(), masks.cpu().float())


def wan_vace_to_video(inputs: dict[str, Any]) -> tuple:
    """WanVaceToVideo — VACE conditioning for MEGA I2V."""
    import torch
    import comfy.utils
    import comfy.model_management
    import node_helpers

    positive = inputs["positive"]
    negative = inputs["negative"]
    vae = inputs["vae"]
    width = int(inputs["width"])
    height = int(inputs["height"])
    length = int(inputs["length"])
    batch_size = int(inputs.get("batch_size", 1))
    strength = float(inputs.get("strength", 1.0))
    control_video = inputs.get("control_video")
    control_masks = inputs.get("control_masks")
    reference_image = inputs.get("reference_image")

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
        import comfy.latent_formats

        reference_image = torch.cat(
            [
                reference_image,
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image)),
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
    out_latent = {"samples": latent}
    return (positive, negative, out_latent, trim_latent)


def register_i2v_node_handlers() -> None:
    from externals.comfy_inprocess.executor import register_node_handler

    register_node_handler("PrimitiveInt", primitive_int)
    register_node_handler("WanVideoVACEStartToEndFrame", wan_vace_start_to_end_frame)
    register_node_handler("WanVaceToVideo", wan_vace_to_video)
