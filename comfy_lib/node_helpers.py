import hashlib
import logging

import torch

from comfy.cli_args import args

from PIL import ImageFile, UnidentifiedImageError


def conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val
            n[1][k] = val
        c.append(n)
    return c


def conditioning_set_values_with_timestep_range(
    conditioning, values={}, start_percent=0.0, end_percent=1.0
):
    """Apply values only during [start_percent, end_percent]; keep conditioning outside."""
    if start_percent > end_percent:
        logging.warning(
            "start_percent (%s) must be <= end_percent (%s)",
            start_percent,
            end_percent,
        )
        return conditioning

    eps = 1e-5
    c = []
    for t in conditioning:
        cond_start = t[1].get("start_percent", 0.0)
        cond_end = t[1].get("end_percent", 1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end = min(end_percent, cond_end)

        if intersect_start >= intersect_end:
            c.append(t)
            continue

        if intersect_start > cond_start:
            c.extend(
                conditioning_set_values(
                    [t],
                    {
                        "start_percent": cond_start,
                        "end_percent": intersect_start - eps,
                    },
                )
            )

        c.extend(
            conditioning_set_values(
                [t],
                {
                    **values,
                    "start_percent": intersect_start,
                    "end_percent": intersect_end,
                },
            )
        )

        if intersect_end < cond_end:
            c.extend(
                conditioning_set_values(
                    [t],
                    {
                        "start_percent": intersect_end + eps,
                        "end_percent": cond_end,
                    },
                )
            )
    return c


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher():
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }
    return hashfuncs[args.default_hashing_function]


def string_to_torch_dtype(string):
    if string == "fp32":
        return torch.float32
    if string == "fp16":
        return torch.float16
    if string == "bf16":
        return torch.bfloat16


def image_alpha_fix(destination, source):
    if destination.shape[-1] < source.shape[-1]:
        source = source[..., : destination.shape[-1]]
    elif destination.shape[-1] > source.shape[-1]:
        source = torch.nn.functional.pad(source, (0, 1))
        source[..., -1] = 1.0
    return destination, source
