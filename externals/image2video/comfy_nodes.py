"""Anthill-only Comfy node handlers (nodes not in comfy_lib NODE_CLASS_MAPPINGS)."""

from __future__ import annotations

from typing import Any


def wan_vace_start_to_end_frame(inputs: dict[str, Any]) -> tuple:
    """WanVideoVACEStartToEndFrame — fallback when ComfyUI-WanVideoWrapper did not load."""
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
        return (control_images, torch.zeros_like(control_images[:, :, :, 0]))

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

    return (out_batch, masks)


def register_i2v_node_handlers() -> None:
    """Register handlers only when WanVideoWrapper (or comfy_extras) did not provide the node."""
    import nodes
    from externals.comfy_inprocess.executor import register_node_handler

    if "WanVideoVACEStartToEndFrame" not in nodes.NODE_CLASS_MAPPINGS:
        register_node_handler("WanVideoVACEStartToEndFrame", wan_vace_start_to_end_frame)
