"""FLUX / SDXL / SD1 / SD3 image generation backends."""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Callable

import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from transformers import CLIPTextModel, T5EncoderModel

from externals.image.model_paths import (
    FLUX_CKPT_4BIT_ID,
    FLUX_CKPT_ID,
    SD3_TURBO_CKPT_ID,
    load_pretrained_sub,
    resolve_lora,
    resolve_model_path,
    resolve_pretrained_dir,
    subfolder_path,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DEFAULT_NEG_PROMPT = (
    "watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, "
    "mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, "
    "disconnected limbs, disconnected head, malformed hands, long neck, mutated hands "
    "and fingers, bad hands, missing fingers, cropped, worst quality, low quality, "
    "mutation, poorly drawn, huge calf, fused hand, missing hand, disappearing arms, "
    "disappearing thigh, disappearing calf, disappearing legs, fused fingers, "
    "abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers"
)

GEN_DISPATCH = ("flux", "flux_ext", "pony", "sd1", "sd3", "sd3cp")


def _compel_classes():
    """Lazy import — only pony/sd1 models need compel (optional extra: uv sync --extra image)."""
    from compel import Compel, ReturnedEmbeddingsType

    return Compel, ReturnedEmbeddingsType


class ImageGen:
    """Generate images for one configured model profile (gentype + checkpoints)."""

    def __init__(
        self,
        name: str,
        gentype: str,
        lora: str = "",
        loras: list | None = None,
        textual_inversions: list | None = None,
        model: str = "",
        poshelp: str = "",
        neghelp: str = "",
        height: int = 512,
        width: int = 768,
        steps: int = 50,
        desc: str = "",
        examples: list | None = None,
        control: str = "",
        lora_weights: list | None = None,
        use_compel: bool = False,
    ):
        self.name = name
        self.gentype = gentype
        self.lora = lora
        self.loras = loras or []
        self.lora_weights = lora_weights
        self.textual_inversions = textual_inversions or []
        self.poshelp = poshelp
        self.neghelp = neghelp
        self.examples = examples or []
        self.height = height
        self.width = width
        self.steps = steps
        self.desc = desc
        self.model = model
        self.control = control
        self.control_texts: list[str] = []
        self.use_compel = use_compel
        self.device: torch.device | None = None
        self.current_loras: list[str] = []

    # --- device / memory -------------------------------------------------------

    def first_init(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.flush()

    def flush(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

    # --- shared helpers ----------------------------------------------------------

    def _checkpoint_id(self, default: str) -> str:
        return resolve_pretrained_dir(self.model or default)

    def _resolve_file(self, ref: str) -> str:
        return resolve_model_path(ref)

    def _random_seed(self, seed: int) -> int:
        return seed if seed else random.randint(0, 999999)

    def _build_prompt(self, prompt: str) -> str:
        if not self.poshelp:
            return prompt
        return f"{prompt}, {self.poshelp}"

    def _build_negative(self, neg: str) -> str:
        parts = [DEFAULT_NEG_PROMPT]
        if neg:
            parts.append(neg)
        if self.neghelp:
            parts.append(self.neghelp)
        # dedupe comma-separated tags
        tags: list[str] = []
        seen: set[str] = set()
        for chunk in ", ".join(parts).split(","):
            tag = chunk.strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return ", ".join(tags)

    def _frame_path(self, filename_pref: str, index: int) -> str:
        return f"{filename_pref}image_{index}.png"

    def _save_frame(self, image, filename_pref: str, index: int) -> str:
        path = self._frame_path(filename_pref, index)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        image.save(path)
        return path

    def _load_loras(self, pipeline, *, weights: list | None = None) -> None:
        if self.lora:
            lora_path, weight_name = resolve_lora(self.lora)
            pipeline.load_lora_weights(lora_path, weight_name=weight_name)
        if not self.current_loras:
            return
        adapters: list[str] = []
        for lr in self.current_loras:
            lora_path, weight_name = resolve_lora(lr)
            adapter_name = Path(lr).stem.replace(".", "_")
            print(f"LOAD_LORA: {lora_path} ({weight_name})")
            pipeline.load_lora_weights(
                lora_path, weight_name=weight_name, adapter_name=adapter_name
            )
            adapters.append(adapter_name)
        adapter_weights = weights if weights is not None else [1.0] * len(adapters)
        pipeline.set_adapters(adapters, adapter_weights=adapter_weights)

    def _run_batch(
        self,
        count: int,
        filename_pref: str,
        generate_one: Callable[[], object],
        *,
        start_index: int = 0,
    ) -> list[str]:
        paths: list[str] = []
        for i in range(count):
            image = generate_one()
            paths.append(self._save_frame(image, filename_pref, start_index + i))
        return paths

    # --- FLUX (two-phase: encode prompt, then generate) --------------------------

    def _flux_encode_prompt(self, ckpt_id: str, ckpt_4bit_id: str, prompt: str):
        """Phase 1: load text encoders only and encode the prompt."""
        ckpt_id = resolve_pretrained_dir(ckpt_id)
        ckpt_4bit_id = resolve_pretrained_dir(ckpt_4bit_id)
        text_encoder_2 = load_pretrained_sub(
            T5EncoderModel, ckpt_4bit_id, "text_encoder_2"
        )
        encode_pipe = FluxPipeline.from_pretrained(
            ckpt_id,
            text_encoder_2=text_encoder_2,
            transformer=None,
            vae=None,
            torch_dtype=torch.float16,
            **({"token": HF_TOKEN} if HF_TOKEN else {}),
        ).to(self.device)

        with torch.no_grad():
            embeds = encode_pipe.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=256
            )
        del encode_pipe
        self.flush()
        return embeds  # prompt_embeds, pooled_prompt_embeds, text_ids

    def _flux_load_transformer(self, ckpt_4bit_id: str, *, custom_checkpoint: bool):
        if custom_checkpoint and self.model:
            return FluxTransformer2DModel.from_single_file(
                self._resolve_file(self.model),
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN or None,
            ).to("cuda")
        return load_pretrained_sub(
            FluxTransformer2DModel, ckpt_4bit_id, "transformer"
        )

    def _flux_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        *,
        custom_transformer: bool = False,
        start_index: int = 0,
    ) -> list[str]:
        """Phase 2: load transformer + CLIP, attach LoRAs, generate frames."""
        ckpt_id = self._checkpoint_id(FLUX_CKPT_ID)
        ckpt_4bit_id = resolve_pretrained_dir(FLUX_CKPT_4BIT_ID)
        seed = self._random_seed(seed)

        prompt_embeds, pooled_prompt_embeds, _text_ids = self._flux_encode_prompt(
            ckpt_id, ckpt_4bit_id, prompt
        )

        text_encoder_1 = CLIPTextModel.from_pretrained(
            subfolder_path(FLUX_CKPT_ID, "text_encoder")
        )
        transformer = self._flux_load_transformer(
            ckpt_4bit_id, custom_checkpoint=custom_transformer
        )

        pipe = FluxPipeline.from_pretrained(
            ckpt_id,
            text_encoder=text_encoder_1,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            transformer=transformer,
            torch_dtype=torch.float16,
            **({"token": HF_TOKEN} if HF_TOKEN else {}),
        ).to(self.device)

        self._load_loras(pipe)
        pipe.enable_model_cpu_offload()

        gen_device = "cuda" if self.device and self.device.type == "cuda" else "cpu"
        variant = [0]

        def _generate():
            run_seed = seed + variant[0]
            variant[0] += 1
            generator = torch.Generator(device=gen_device).manual_seed(run_seed)
            return pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=self.steps,
                guidance_scale=5.5,
                height=self.height,
                width=self.width,
                output_type="pil",
                generator=generator,
            ).images[0]

        paths = self._run_batch(count, filename_pref, _generate, start_index=start_index)
        del pipe
        self.flush()
        return paths

    def flux_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        *,
        start_index: int = 0,
    ) -> list[str]:
        return self._flux_gen_images(
            prompt, filename_pref, count, seed,
            custom_transformer=False, start_index=start_index,
        )

    def flux_ext_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        *,
        start_index: int = 0,
    ) -> list[str]:
        return self._flux_gen_images(
            prompt, filename_pref, count, seed,
            custom_transformer=True, start_index=start_index,
        )

    # --- SDXL (pony) -------------------------------------------------------------

    def pony_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        neg: str = "",
        lora_weights: list | None = None,
        *,
        start_index: int = 0,
    ) -> list[str]:
        model_ref = self._resolve_file(self.model) if self.model else self.model
        if model_ref and ".safetensors" in model_ref:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_ref, torch_dtype=torch.float16, added_cond_kwargs={}
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                resolve_pretrained_dir(self.model),
                torch_dtype=torch.float16,
                added_cond_kwargs={},
            )
        pipe = pipe.to("cuda")
        self._load_loras(pipe, weights=lora_weights)

        prompt_embeds = pooled_prompt_embeds = None
        if self.use_compel:
            Compel, ReturnedEmbeddingsType = _compel_classes()
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)

        neg_prompt = self._build_negative(neg)
        print(f"pt: {prompt}")
        print(f"ng: {neg_prompt}")

        def _generate():
            kwargs = dict(
                negative_prompt=neg_prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.steps,
            )
            if self.use_compel:
                return pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    **kwargs,
                ).images[0]
            return pipe(prompt, **kwargs).images[0]

        paths = self._run_batch(count, filename_pref, _generate, start_index=start_index)
        del pipe
        self.flush()
        return paths

    # --- SD 1.x ------------------------------------------------------------------

    def sd1_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        neg: str = "",
        *,
        start_index: int = 0,
    ) -> list[str]:
        model_ref = self._resolve_file(self.model) if self.model else self.model
        if model_ref and ".safetensors" in model_ref:
            pipe = StableDiffusionPipeline.from_single_file(
                model_ref, torch_dtype=torch.float16, added_cond_kwargs={}
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                resolve_pretrained_dir(self.model),
                torch_dtype=torch.float16,
                added_cond_kwargs={},
            )
        pipe = pipe.to("cuda")
        self._load_loras(pipe)

        neg_prompt = neg or DEFAULT_NEG_PROMPT
        prompt_embeds = None
        if self.use_compel:
            Compel, _ReturnedEmbeddingsType = _compel_classes()
            compel = Compel(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                requires_pooled=False,
            )
            prompt_embeds = compel(prompt)

        def _generate():
            kwargs = dict(
                negative_prompt=neg_prompt,
                num_inference_steps=self.steps,
                height=self.height,
                width=self.width,
            )
            if self.use_compel:
                return pipe(prompt_embeds=prompt_embeds, **kwargs).images[0]
            return pipe(prompt, **kwargs).images[0]

        paths = self._run_batch(count, filename_pref, _generate, start_index=start_index)
        del pipe
        self.flush()
        return paths

    # --- SD 3.x ------------------------------------------------------------------

    def _sd3_pipe(self):
        model_ref = self._resolve_file(self.model) if self.model else self.model
        if model_ref and ".safetensors" in model_ref:
            return StableDiffusion3Pipeline.from_single_file(
                model_ref, torch_dtype=torch.float16, added_cond_kwargs={}
            )
        return StableDiffusion3Pipeline.from_pretrained(
            resolve_pretrained_dir(self.model),
            torch_dtype=torch.float16,
            added_cond_kwargs={},
        )

    def sd3_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        *,
        start_index: int = 0,
    ) -> list[str]:
        pipe = self._sd3_pipe().to("cuda")
        self._load_loras(pipe)

        def _generate():
            return pipe(
                prompt,
                negative_prompt=DEFAULT_NEG_PROMPT,
                num_inference_steps=self.steps,
            ).images[0]

        paths = self._run_batch(count, filename_pref, _generate, start_index=start_index)
        del pipe
        self.flush()
        return paths

    def sd3_checkpoint_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 5,
        seed: int = 0,
        *,
        start_index: int = 0,
    ) -> list[str]:
        transformer = SD3Transformer2DModel.from_single_file(
            self._resolve_file(self.model), torch_dtype=torch.bfloat16
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            resolve_pretrained_dir(SD3_TURBO_CKPT_ID),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            added_cond_kwargs={},
        ).to("cuda")

        def _generate():
            return pipe(
                prompt,
                negative_prompt=DEFAULT_NEG_PROMPT,
                num_inference_steps=self.steps,
            ).images[0]

        paths = self._run_batch(count, filename_pref, _generate, start_index=start_index)
        del pipe
        self.flush()
        return paths

    # --- entry point -------------------------------------------------------------

    def gen(
        self,
        prompt: str,
        seed: int = 0,
        count: int = 1,
        sfx: str = "",
        neg: str = "",
        prfx: str = "",
        height: int = 0,
        width: int = 0,
        add_concepts: list | None = None,
        output_dir: str | None = None,
        start_index: int = 0,
        file_prefix: str = "",
    ) -> list[str]:
        self.current_loras = self.loras + (add_concepts or [])
        saved_height, saved_width = self.height, self.width
        out_height = height if height > 0 else self.height
        out_width = width if width > 0 else self.width
        self.height, self.width = out_height, out_width

        seed = self._random_seed(seed)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename_pref = output_dir.rstrip("/\\") + "/" + file_prefix
        else:
            filename_pref = (
                f"frames/{prfx}{file_prefix}{self.gentype}_{self.name}_{sfx}"
            )
        final_prompt = self._build_prompt(prompt)
        print(final_prompt)

        self.first_init()

        try:
            if self.gentype == "flux":
                return self.flux_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, start_index=start_index,
                )
            if self.gentype == "flux_ext":
                return self.flux_ext_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, start_index=start_index,
                )
            if self.gentype == "pony":
                return self.pony_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, neg=neg,
                    lora_weights=self.lora_weights, start_index=start_index,
                )
            if self.gentype == "sd1":
                return self.sd1_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, neg=neg,
                    start_index=start_index,
                )
            if self.gentype == "sd3":
                return self.sd3_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, start_index=start_index,
                )
            if self.gentype == "sd3cp":
                return self.sd3_checkpoint_gen_images(
                    final_prompt, filename_pref, count=count, seed=seed, start_index=start_index,
                )
            raise ValueError(f"Unknown gentype: {self.gentype!r} (expected one of {GEN_DISPATCH})")
        finally:
            self.height, self.width = saved_height, saved_width

