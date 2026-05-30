"""Tests for Comfy-matched $image2image sampling helpers."""

from __future__ import annotations

import unittest

import torch

from externals.image2image.comfy_nodes import (
    DEFAULT_TARGET_SIZE,
    ModelSamplingDiscreteFlow,
    build_comfy_image_prompt,
    comfy_beta_sigmas,
    vae_reference_dimensions,
    vl_dimensions,
)
from externals.image2image.comfy_sampler import FlowModelSampling, sample_sa_solver
from externals.image2image.qwen_pipeline import (
    COMFY_VAE_REFERENCE_AREA,
    _comfy_scheduler_config,
    base_assets_ready,
    base_model_dir,
    should_full_gpu,
)


class TestComfyNodes(unittest.TestCase):
    def test_vl_area_is_384_squared(self) -> None:
        w, h = vl_dimensions(1920, 1080)
        self.assertAlmostEqual(w * h, 384 * 384, delta=500)

    def test_vae_reference_uses_floor_snap(self) -> None:
        w, h = vae_reference_dimensions(1000, 1000, target_size=896)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertLessEqual(w * h, 896 * 896 + 32 * 896)

    def test_picture_prompt_prefix(self) -> None:
        text = build_comfy_image_prompt(2)
        self.assertIn("Picture 1:", text)
        self.assertIn("Picture 2:", text)

    def test_comfy_beta_sigmas_end_at_zero(self) -> None:
        sigmas = comfy_beta_sigmas(4, shift=1.0, device="cpu")
        self.assertEqual(len(sigmas), 5)
        self.assertEqual(sigmas[-1].item(), 0.0)


class TestComfySampler(unittest.TestCase):
    def test_flow_model_sampling_percent_to_sigma(self) -> None:
        sampling = FlowModelSampling(shift=1.0)
        self.assertAlmostEqual(sampling.percent_to_sigma(0.0), 1.0)
        self.assertAlmostEqual(sampling.percent_to_sigma(1.0), 0.0)
        self.assertAlmostEqual(sampling.percent_to_sigma(0.5), 0.5)

    def test_sa_solver_runs_on_toy_tensors(self) -> None:
        x = torch.zeros(1, 4, 2)
        sigmas = torch.tensor([1.0, 0.5, 0.0])

        class _ToyModel:
            def __call__(self, latent, _sigma, **_extra):
                return latent

        out = sample_sa_solver(_ToyModel(), x, sigmas, model_sampling=FlowModelSampling())
        self.assertEqual(out.shape, x.shape)

    def test_sa_solver_matches_bfloat16_latents(self) -> None:
        x = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.0])

        class _ToyModel:
            def __call__(self, latent, _sigma, **_extra):
                return latent

        out = sample_sa_solver(_ToyModel(), x, sigmas, model_sampling=ModelSamplingDiscreteFlow())
        self.assertEqual(out.dtype, torch.bfloat16)


class TestQwenPipelineTuning(unittest.TestCase):
    def test_vae_reference_area_matches_comfy(self) -> None:
        self.assertEqual(COMFY_VAE_REFERENCE_AREA, DEFAULT_TARGET_SIZE * DEFAULT_TARGET_SIZE)

    def test_scheduler_leaves_diffusers_beta_disabled(self) -> None:
        if not base_assets_ready():
            self.skipTest("Qwen base assets not installed")
        config = _comfy_scheduler_config(str(base_model_dir()))
        self.assertFalse(config.get("use_beta_sigmas"))

    def test_should_full_gpu_respects_cpu(self) -> None:
        self.assertFalse(should_full_gpu(use_gpu=False))


if __name__ == "__main__":
    unittest.main()
