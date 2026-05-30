"""Comfy-compatible beta + SA-Solver sampling for Qwen flow-match edit."""

from __future__ import annotations

import math
from typing import Callable

import torch

from externals.image2image.comfy_nodes import ModelSamplingDiscreteFlow
from externals.image2image import sa_solver


def time_snr_shift(alpha: float, t: float | torch.Tensor) -> float | torch.Tensor:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


# Back-compat alias for tests / SA-Solver tau defaults.
FlowModelSampling = ModelSamplingDiscreteFlow


def sigma_to_half_log_snr(sigmas: torch.Tensor) -> torch.Tensor:
    return sigmas.logit().neg()


def offset_first_sigma_for_snr(
    sigmas: torch.Tensor,
    model_sampling: FlowModelSampling,
    percent_offset: float = 1e-4,
) -> torch.Tensor:
    if len(sigmas) <= 1:
        return sigmas
    if sigmas[0].item() >= 1:
        sigmas = sigmas.clone()
        sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas


def default_noise_sampler(x: torch.Tensor, seed: int | None = None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    def _sample(_sigma, _sigma_next):
        return torch.randn(
            x.size(),
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
            generator=generator,
        )

    return _sample


def _append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
    while value.ndim < ndim:
        value = value.unsqueeze(-1)
    return value


@torch.no_grad()
def sample_sa_solver(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    extra_args: dict | None = None,
    tau_func: Callable | None = None,
    s_noise: float = 1.0,
    noise_sampler=None,
    predictor_order: int = 3,
    corrector_order: int = 4,
    use_pece: bool = False,
    simple_order_2: bool = False,
    model_sampling: FlowModelSampling | None = None,
) -> torch.Tensor:
    """Stochastic Adams Solver (Comfy sa_solver + beta sigmas)."""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    if model_sampling is None:
        model_sampling = FlowModelSampling()
    s_noise = s_noise * model_sampling.noise_scale
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas)

    if tau_func is None:
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x
    h = 0.0
    tau_t = 0.0
    noise = torch.zeros_like(x)
    pred_list: list[torch.Tensor] = []
    lower_order_to_end = sigmas[-1].item() == 0

    for i in range(len(sigmas) - 1):
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1 : i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)
            b_coeffs = b_coeffs.to(dtype=pred_mat.dtype)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            ratio = (sigmas[i] / sigmas[i - 1] * (-(tau_t**2) * h).exp()).to(dtype=x.dtype)
            x = ratio * x + corr_res

            if tau_t > 0 and s_noise > 0:
                x = x + noise

        if use_pece:
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            pred_list[-1] = denoised

        if sigmas[i + 1] == 0:
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1].item())
            curr_lambdas = lambdas[i - predictor_order_used + 1 : i + 1]
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)
            b_coeffs = b_coeffs.to(dtype=pred_mat.dtype)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            ratio = (sigmas[i + 1] / sigmas[i] * (-(tau_t**2) * h).exp()).to(dtype=x.dtype)
            x_pred = ratio * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise_scale = (
                    sigmas[i + 1]
                    * (-2 * tau_t**2 * h).expm1().neg().sqrt()
                    * s_noise
                ).to(dtype=x.dtype)
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale
                x_pred = x_pred + noise

    return x_pred


class QwenFlowDenoiser:
    """Wrap diffusers Qwen transformer as a Comfy-style denoiser (returns x0)."""

    def __init__(
        self,
        pipe,
        *,
        image_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes,
        guidance: torch.Tensor | None,
        latent_len: int,
    ) -> None:
        self.pipe = pipe
        self.image_latents = image_latents
        self.prompt_embeds = prompt_embeds
        self.prompt_embeds_mask = prompt_embeds_mask
        self.img_shapes = img_shapes
        self.guidance = guidance
        self.latent_len = latent_len

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **_extra) -> torch.Tensor:
        latent_model_input = x
        if self.image_latents is not None:
            latent_model_input = torch.cat([x, self.image_latents], dim=1)

        if sigma.ndim == 0:
            timestep = (sigma * 1000).expand(x.shape[0]).to(dtype=x.dtype, device=x.device)
        else:
            timestep = (sigma * 1000).to(dtype=x.dtype, device=x.device)

        with self.pipe.transformer.cache_context("cond"):
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=self.guidance,
                encoder_hidden_states_mask=self.prompt_embeds_mask,
                encoder_hidden_states=self.prompt_embeds,
                img_shapes=self.img_shapes,
                attention_kwargs=self.pipe.attention_kwargs or {},
                return_dict=False,
            )[0]
        noise_pred = noise_pred[:, : self.latent_len]
        sigma_b = _append_dims(sigma.to(dtype=x.dtype, device=x.device), x.ndim)
        return x - sigma_b * noise_pred
