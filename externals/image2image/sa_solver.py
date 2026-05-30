# SA-Solver: Stochastic Adams Solver (NeurIPS 2023, arXiv:2309.05019)
# Vendored from ComfyUI comfy/k_diffusion/sa_solver.py (MIT).

from __future__ import annotations

import math
from typing import Callable, Union

import torch


def compute_exponential_coeffs(
    s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float
) -> torch.Tensor:
    tau_mul = 1 + tau_t**2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms_factored = t**p - s**p * (-tau_mul * h).exp()
    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()
    return recursive_coeff_mat @ product_terms_factored


def compute_simple_stochastic_adams_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    is_corrector_step: bool = False,
) -> torch.Tensor:
    tau_mul = 1 + tau_t**2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        b_2 = alpha_t * (0.5 * tau_mul * h**2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])


def compute_stochastic_adams_b_coeffs(
    sigma_next: torch.Tensor,
    curr_lambdas: torch.Tensor,
    lambda_s: torch.Tensor,
    lambda_t: torch.Tensor,
    tau_t: float,
    simple_order_2: bool = False,
    is_corrector_step: bool = False,
) -> torch.Tensor:
    num_timesteps = curr_lambdas.shape[0]
    if simple_order_2 and num_timesteps == 2:
        return compute_simple_stochastic_adams_b_coeffs(
            sigma_next, curr_lambdas, lambda_s, lambda_t, tau_t, is_corrector_step
        )

    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_t = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_t, exp_integral_coeffs)
    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals


def get_tau_interval_func(
    start_sigma: float, end_sigma: float, eta: float = 1.0
) -> Callable[[Union[torch.Tensor, float]], float]:
    def tau_func(sigma: Union[torch.Tensor, float]) -> float:
        if eta <= 0:
            return 0.0
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return eta if start_sigma >= sigma >= end_sigma else 0.0

    return tau_func
