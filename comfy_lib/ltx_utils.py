
import comfy.model_sampling
import comfy.samplers
import node_helpers
import folder_paths
import latent_preview
from comfy.samplers import KSAMPLER
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy.utils import common_upscale

#import io
#import nodes
#import comfy.model_management
#import comfy.model_sampling
#import comfy.utils
import io
from typing import Union
import torch
from torch import Tensor
import imageio
import cv2
import itertools
from PIL import Image, ImageOps, ImageSequence
import math
import numpy as np
import av
from tqdm import trange

"""
return {"required": {"positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "vae": ("VAE",),
                     "latent": ("LATENT",),
                     "image": ("IMAGE", {"tooltip": "Image or video to condition the latent video on. Must be 8*n + 1 frames."
                                         "If the video is not 8*n + 1 frames, it will be cropped to the nearest 8*n + 1 frames."}),
                     "frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999,
                                           "tooltip": "Frame index to start the conditioning at. For single-frame images or "
                                           "videos with 1-8 frames, any frame_idx value is acceptable. For videos with 9+ "
                                           "frames, frame_idx must be divisible by 8, otherwise it will be rounded down to "
                                           "the nearest multiple of 8. Negative values are counted from the end of the video."}),
                     "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
    }
"""


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask

def get_keyframe_idxs(cond):
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    num_keyframes = torch.unique(keyframe_idxs[:, 0]).shape[0]
    return keyframe_idxs, num_keyframes

def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


class SimpleLTXVAddGuide:
    def __init__(self):
        self._num_prefix_frames = 2
        self._patchifier = SymmetricPatchifier(1)

    def encode(self, vae, latent_width, latent_height, images, scale_factors):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
        pixels = comfy.utils.common_upscale(images.movedim(-1, 1), latent_width * width_scale_factor, latent_height * height_scale_factor, "bilinear", crop="disabled").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return encode_pixels, t

    def get_latent_index(self, cond, latent_length, guide_length, frame_idx, scale_factors):
        time_scale_factor, _, _ = scale_factors
        _, num_keyframes = get_keyframe_idxs(cond)
        latent_count = latent_length - num_keyframes
        frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
        if guide_length > 1:
            frame_idx = frame_idx // time_scale_factor * time_scale_factor # frame index must be divisible by 8

        latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor

        return frame_idx, latent_idx

    def add_keyframe_index(self, cond, frame_idx, guiding_latent, scale_factors):
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = self._patchifier.patchify(guiding_latent)
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, True)
        pixel_coords[:, 0] += frame_idx
        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    def append_keyframe(self, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors):
        positive = self.add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors)
        negative = self.add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors)

        mask = torch.full(
            (noise_mask.shape[0], 1, guiding_latent.shape[2], 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = torch.cat([latent_image, guiding_latent], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)
        return positive, negative, latent_image, noise_mask

    def replace_latent_frames(self, latent_image, noise_mask, guiding_latent, latent_idx, strength):
        cond_length = guiding_latent.shape[2]
        assert latent_image.shape[2] >= latent_idx + cond_length, "Conditioning frames exceed the length of the latent sequence."

        mask = torch.full(
            (noise_mask.shape[0], 1, cond_length, 1, 1),
            1.0 - strength,
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = latent_image.clone()
        noise_mask = noise_mask.clone()

        latent_image[:, :, latent_idx : latent_idx + cond_length] = guiding_latent
        noise_mask[:, :, latent_idx : latent_idx + cond_length] = mask

        return latent_image, noise_mask

    def generate(self, positive, negative, vae, latent, image, frame_idx, strength):
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        image, t = self.encode(vae, latent_width, latent_height, image, scale_factors)

        frame_idx, latent_idx = self.get_latent_index(positive, latent_length, len(image), frame_idx, scale_factors)
        assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

        num_prefix_frames = min(self._num_prefix_frames, t.shape[2])

        positive, negative, latent_image, noise_mask = self.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            t[:, :, :num_prefix_frames],
            strength,
            scale_factors,
        )

        latent_idx += num_prefix_frames

        t = t[:, :, num_prefix_frames:]
        if t.shape[2] == 0:
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

        latent_image, noise_mask = self.replace_latent_frames(
            latent_image,
            noise_mask,
            t,
            latent_idx,
            strength,
        )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

def load_image(image_path):
    #image_path = folder_paths.get_annotated_filepath(image)

    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)



def LTXV_Conditioning(positive, negative, frame_rate):
    positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
    negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
    return (positive, negative)

def clip_encode(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)

def empty_LTXV_latent_video(width=768, height=512, length=97, batch_size=1):
    latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
    return {"samples": latent}

def get_sigmas(steps=20, max_shift=2.05, base_shift=0.95, stretch=False, terminal=0.10, latent=None):
    if latent is None:
        tokens = 4096
    else:
        tokens = math.prod(latent["samples"].shape[2:])

    sigmas = torch.linspace(1.0, 0.0, steps + 1)

    x1 = 1024
    x2 = 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = (tokens) * mm + b

    power = 1
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
        0,
    )

    # Stretch sigmas so that its final value matches the given terminal value.
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched

    return sigmas


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


def get_sampler(sampler_name):
    sampler = comfy.samplers.sampler_object(sampler_name)
    return sampler

#def sampler_custom(model, add_noise=True, noise_seed=0, cfg=8.0, positive, negative, sampler_name='euler_ancestral', latent, latent_image):
def sampler_custom(model, add_noise, noise_seed, cfg, positive, negative, sampler_name, sigmas, latent_image, sampler=None):
    if sampler is None:
        sampler = get_sampler(sampler_name)
    #sigmas = get_sigmas(latent=latent_image)

    latent = latent_image
    latent_image = latent["samples"]
    latent = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    latent["samples"] = latent_image

    if not add_noise:
        noise = Noise_EmptyNoise().generate_noise(latent)
    else:
        noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

    out = latent.copy()
    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return (out, out_denoised)



DEFAULT_PAG_LTX = {"layers": set([14])}
def LTX_Perturbed_Attention(
    model, scale=2.0, rescale=0.5, cfg=3.0, attn_override=DEFAULT_PAG_LTX, attn_type="PAG"
):
    m = model.clone()

    def pag_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
        return v

    def seg_fn(q, k, v, heads, attn_precision=None, transformer_options=None):
        _, sequence_length, _ = q.shape
        b, c, f, h, w = transformer_options["original_shape"]

        q = rearrange(q, "b (f h w) d -> b (f d) w h", h=h, w=w)
        kernel_size = math.ceil(6 * scale) + 1 - math.ceil(6 * scale) % 2
        q = gaussian_blur_2d(q, kernel_size, scale)
        q = rearrange(q, "b (f d) w h -> b (f h w) d", f=f)
        return optimized_attention(q, k, v, heads, attn_precision=attn_precision)

    def post_cfg_function(args):
        model = args["model"]

        cond_pred = args["cond_denoised"]
        uncond_pred = args["uncond_denoised"]

        len_conds = 1 if args.get("uncond", None) is None else 2

        cond = args["cond"]
        sigma = args["sigma"]
        model_options = args["model_options"].copy()
        x = args["input"]

        if scale == 0:
            if len_conds == 1:
                return cond_pred
            return uncond_pred + (cond_pred - uncond_pred)

        attn_fn = pag_fn if attn_type == "PAG" else seg_fn
        for block_idx in attn_override["layers"]:
            model_options = comfy.model_patcher.set_model_options_patch_replace(
                model_options, attn_fn, "layer", "self_attn", int(block_idx)
            )

        (perturbed,) = comfy.samplers.calc_cond_batch(
            model, [cond], x, sigma, model_options
        )

        # if len_conds == 1:
        #     output = cond_pred + scale * (cond_pred - pag)
        # else:
        #     output = cond_pred + (scale-1.0) * (cond_pred - uncond_pred) + scale * (cond_pred - pag)

        output = (
            uncond_pred
            + cfg * (cond_pred - uncond_pred)
            + scale * (cond_pred - perturbed)
        )
        if rescale > 0:
            factor = cond_pred.std() / output.std()
            factor = rescale * factor + (1 - rescale)
            output = output * factor

        return output

    m.set_model_sampler_post_cfg_function(post_cfg_function)

    return m

def ltxv_crop_guides(positive, negative, latent):
    latent_image = latent["samples"].clone()
    noise_mask = get_noise_mask(latent)

    _, num_keyframes = get_keyframe_idxs(positive)
    if num_keyframes == 0:
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

    latent_image = latent_image[:, :, :-num_keyframes]
    noise_mask = noise_mask[:, :, :-num_keyframes]

    positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
    negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

    return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)


def vae_decode_tiles(vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    compression = vae.spacial_compression_decode()
    images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)


def save_video(filename, images, fps=25):
    writer = imageio.get_writer(filename, fps=fps)
    for img in images:
        writer.append_data(tensor_to_bytes(img))
    writer.close()

class AttentionBank:
    def __init__(self, save_steps, block_map, inject_steps=None):
        self._data = {
            "save_steps": save_steps,
            "block_map": block_map,
            "inject_steps": inject_steps,
        }

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

def create_attn_bank(save_steps, blocks=""):
    block_map = {}
    block_list = blocks.split(",")
    for block in block_list:
        block_idx = int(block)
        block_map[block_idx] = {}

    bank = AttentionBank(save_steps, block_map)
    return bank

def vae_encode(vae, pixels):
    t = vae.encode(pixels[:,:,:,:3])
    return {"samples":t}



def LTX_prepare_attn_injections(latent, attn_bank, query, key, value, inject_steps, blocks=None):
        if inject_steps > attn_bank["save_steps"]:
            raise ValueError("Can not inject more steps than were saved.")
        attn_bank = AttentionBank(
            attn_bank["save_steps"], attn_bank["block_map"], inject_steps
        )
        attn_bank["inject_settings"] = set([])
        if query:
            attn_bank["inject_settings"].add("q")
        if key:
            attn_bank["inject_settings"].add("k")
        if value:
            attn_bank["inject_settings"].add("v")

        if blocks is not None:
            attn_bank["block_map"] = {**attn_bank["block_map"]}
            for key in list(attn_bank["block_map"].keys()):
                if key not in blocks:
                    del attn_bank["block_map"][key]

        # Hack to force order of operations in ComfyUI graph
        return (latent, attn_bank)


class ReverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output  # model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)
def LTX_Reverse_Model_Sampling_Pred(model):
    m = model.clone()

    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = ReverseCONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=1.15)
    m.add_object_patch("model_sampling", model_sampling)
    return m


class InverseCONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_output

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent
def LTX_Forward_Model_Sampling_Pre(model):
    m = model.clone()

    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = InverseCONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model.model.model_config)
    model_sampling.set_parameters(shift=1.15)
    m.add_object_patch("model_sampling", model_sampling)
    return m




def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "h264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()
def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")
def preprocess(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (image[:(image.shape[0] // 2) * 2, :(image.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
    with io.BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor
def LTXV_Preprocess(image, img_compression):
    if img_compression > 0:
        output_images = []
        for i in range(image.shape[0]):
            output_images.append(preprocess(image[i], img_compression))
    return torch.stack(output_images)
    


def resize_image(image, width, height, keep_proportion, upscale_method, divisible_by, 
           width_input=None, height_input=None, get_image_size=None, crop="disabled"):
    B, H, W, C = image.shape

    if width_input:
        width = width_input
    if height_input:
        height = height_input
    if get_image_size is not None:
        _, height, width, _ = get_image_size.shape
    
    if keep_proportion and get_image_size is None:
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            if width == 0 and height != 0:
                ratio = height / H
                width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                height = round(H * ratio)
            elif width != 0 and height != 0:
                # Scale based on which dimension is smaller in proportion to the desired dimensions
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
    else:
        if width == 0:
            width = W
        if height == 0:
            height = H
  
    if divisible_by > 1 and get_image_size is None:
        width = width - (width % divisible_by)
        height = height - (height % divisible_by)
    
    image = image.movedim(-1,1)
    image = common_upscale(image, width, height, upscale_method, crop)
    image = image.movedim(1,-1)

    return(image, image.shape[2], image.shape[1],)



def flip_sigmas(sigmas):
    if len(sigmas) == 0:
        return sigmas

    sigmas = sigmas.flip(0)
    if sigmas[0] == 0:
        sigmas[0] = 0.0001
    return sigmas


