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


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _falsy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "false", "no", "off")


def _gpu_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _file_gib(path: str) -> float:
    p = Path(path)
    if not p.is_file():
        return 0.0
    return p.stat().st_size / (1024**3)


def _flux_ext_needs_cpu_offload(model_ref: str) -> bool:
    if _truthy_env("AH_IMAGE_CPU_OFFLOAD"):
        return True
    if _falsy_env("AH_IMAGE_CPU_OFFLOAD"):
        return False
    path = resolve_model_path(model_ref)
    ckpt_gib = _file_gib(path)
    vram_gib = _gpu_vram_gib()
    if ckpt_gib <= 0 or vram_gib <= 0:
        return True
    # Checkpoint size is mostly transformer weights; leave headroom for VAE + activations.
    return ckpt_gib + 2.5 > vram_gib


DEFAULT_NEG_PROMPT = (
    "watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, "
    "mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, "
    "disconnected limbs, disconnected head, malformed hands, long neck, mutated hands "
    "and fingers, bad hands, missing fingers, cropped, worst quality, low quality, "
    "mutation, poorly drawn, huge calf, fused hand, missing hand, disappearing arms, "
    "disappearing thigh, disappearing calf, disappearing legs, fused fingers, "
    "abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers"
)

GEN_DISPATCH = ("flux", "flux_ext", "flux2_klein", "pony", "sd1", "sd3", "sd3cp")


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
        guidance_scale: float = 5.5,
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
        self.guidance_scale = guidance_scale
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

    def _flux_clip_prompt(self, tokenizer, prompt: str, *, max_length: int = 77) -> str:
        """CLIP pooled embeds are capped at 77 tokens; keep the start of the prompt."""
        if tokenizer is None:
            return prompt
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

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

        clip_prompt = self._flux_clip_prompt(encode_pipe.tokenizer, prompt)
        t5_max_length = 512

        with torch.no_grad():
            embeds = encode_pipe.encode_prompt(
                prompt=clip_prompt,
                prompt_2=prompt,
                max_sequence_length=t5_max_length,
            )
        del encode_pipe
        self.flush()
        return embeds  # prompt_embeds, pooled_prompt_embeds, text_ids

    def _flux_load_transformer(self, ckpt_4bit_id: str, *, custom_checkpoint: bool):
        if custom_checkpoint and self.model:
            path = self._resolve_file(self.model)
            from externals.image.flux_aio_loader import is_comfy_flux_aio, load_flux_aio_transformer

            if is_comfy_flux_aio(path):
                return load_flux_aio_transformer(path)

            transformer_config = subfolder_path(FLUX_CKPT_4BIT_ID, "transformer")
            use_offload = _flux_ext_needs_cpu_offload(self.model)
            if use_offload:
                ckpt_gib = _file_gib(path)
                vram_gib = _gpu_vram_gib()
                print(
                    f"$image: flux_ext checkpoint {ckpt_gib:.1f} GiB exceeds "
                    f"GPU {vram_gib:.1f} GiB — CPU offload (~minutes/image). "
                    "Use an NF4/GGUF checkpoint in models/flux/ for ~10–30 s/image.",
                    flush=True,
                )
            transformer = FluxTransformer2DModel.from_single_file(
                path,
                config=transformer_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=use_offload,
                token=HF_TOKEN or None,
            )
            if not use_offload:
                transformer = transformer.to("cuda")
            return transformer
        return load_pretrained_sub(
            FluxTransformer2DModel, ckpt_4bit_id, "transformer"
        )

    def _flux_deploy_pipeline(self, pipe, *, custom_transformer: bool) -> str:
        """Keep flux_ext on GPU when the checkpoint fits; otherwise use CPU offload."""
        if self.device is None or self.device.type != "cuda":
            pipe.enable_model_cpu_offload()
            return "cpu"
        if custom_transformer and _flux_ext_needs_cpu_offload(self.model):
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.enable_slicing()
            return "cuda"
        if _truthy_env("AH_IMAGE_CPU_OFFLOAD"):
            pipe.enable_model_cpu_offload()
            return "cuda"
        if _falsy_env("AH_IMAGE_CPU_OFFLOAD") or custom_transformer:
            pipe.to("cuda")
            return "cuda"
        pipe.enable_model_cpu_offload()
        return "cuda"

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
        # flux_ext: self.model is a single-file transformer, not the FLUX.1-dev repo.
        if custom_transformer:
            ckpt_id = resolve_pretrained_dir(FLUX_CKPT_ID)
        else:
            ckpt_id = self._checkpoint_id(FLUX_CKPT_ID)
        ckpt_4bit_id = resolve_pretrained_dir(FLUX_CKPT_4BIT_ID)
        seed = self._random_seed(seed)

        prompt_embeds, pooled_prompt_embeds, _text_ids = self._flux_encode_prompt(
            ckpt_id, ckpt_4bit_id, prompt
        )

        text_encoder_1 = None
        if not custom_transformer:
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
        )

        self._load_loras(pipe)
        gen_device = self._flux_deploy_pipeline(
            pipe, custom_transformer=custom_transformer
        )

        variant = [0]

        def _generate():
            run_seed = seed + variant[0]
            variant[0] += 1
            generator = torch.Generator(device=gen_device).manual_seed(run_seed)
            return pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
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

    def flux2_klein_gen_images(
        self,
        prompt: str,
        filename_pref: str,
        count: int = 1,
        seed: int = 0,
        *,
        start_index: int = 0,
    ) -> list[str]:
        from externals.flux2_klein.comfy_runner import (
            run_klein_txt2img,
            run_klein_txt2img_variants,
        )

        work_dir = Path(filename_pref).parent.parent / "comfy_klein"
        if count > 1:
            seeds = [seed + i if seed else None for i in range(count)]
            paths: list[str] = []

            def _on_sample(pil, i: int) -> None:
                path = self._save_frame(pil, filename_pref, start_index + i)
                paths.append(path)
                print(f"$image: wrote {path}", flush=True)

            run_klein_txt2img_variants(
                work_dir=work_dir,
                prompt=prompt,
                model_arg=self.model or "klein-fp8",
                steps=self.steps,
                cfg=self.guidance_scale,
                seeds=seeds,
                width=self.width,
                height=self.height,
                use_gpu=torch.cuda.is_available(),
                on_sample=_on_sample,
            )
            return paths

        run_seed = seed if seed else None
        pil = run_klein_txt2img(
            work_dir=work_dir,
            prompt=prompt,
            model_arg=self.model or "klein-fp8",
            steps=self.steps,
            cfg=self.guidance_scale,
            seed=run_seed,
            width=self.width,
            height=self.height,
            use_gpu=torch.cuda.is_available(),
        )
        path = self._save_frame(pil, filename_pref, start_index)
        print(f"$image: wrote {path}", flush=True)
        return [path]

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
            if self.gentype == "flux2_klein":
                return self.flux2_klein_gen_images(
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

