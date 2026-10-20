# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Reweighted penalized least squares on a batch of independent windows, with a fixed offset.

``AdaptiveTAM`` solves one small system per sliding window. For the base of a non-Gaussian loss each window is a GLM
``eta = offset + Phi theta`` whose offset is the linear predictor of the base model, so the window learns a correction on the link scale.
This is the loop of ``_reweighting.reweighted_penalized_fit`` run on all the windows at once: the working response and the Fisher
weights come from the strategy, the weights are normalised to unit mean per window, and a window whose penalized deviance
rises is step-halved on its own. The dispersion of a negative binomial or Tweedie base is the one of the base model.
"""
from __future__ import annotations

import torch

from tam.model._math import _compute_weighted_covariances, solve_linear_system

from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-12


def batched_reweighted_solve(
    phi: torch.Tensor,
    y: torch.Tensor,
    offset: torch.Tensor,
    strategy: ReweightingStrategy,
    penalty_matrix: torch.Tensor,
    loss_matrix: torch.Tensor,
    num_samples: int,
    max_iter: int = 25,
    tol: float = 1e-6,
    max_halvings: int = 8,
) -> torch.Tensor:
    """Coefficients of every window, shape (n_windows, n_coeffs, 1).

    Args:
        phi: Design matrices, (n_windows, n_samples, n_coeffs).
        y: Responses, (n_windows, n_samples, 1).
        offset: Linear predictor of the base model, (n_windows, n_samples, 1).
        strategy: The GLM family of the base model.
        penalty_matrix: The penalty of the window model.
        loss_matrix: The loss matrix of the window model.
        num_samples: Rows of a window (the scale of the penalty).
        max_iter: Maximum reweighting passes.
        tol: Relative change of the linear predictor, in every window, under which the loop stops.
        max_halvings: Maximum step-halvings per pass and window.
    """
    dtype = phi.dtype
    y = y.to(device=phi.device, dtype=dtype)
    offset = offset.to(device=phi.device, dtype=dtype)
    penalty = penalty_matrix.to(device=phi.device, dtype=dtype)
    loss = loss_matrix.to(device=phi.device)
    unit_deviance = getattr(strategy, "unit_deviance", None)

    def objective(theta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        mean = strategy.inverse_link(eta)
        data_term = (unit_deviance(y, mean) if unit_deviance is not None else (y - mean) ** 2).mean(dim=(1, 2))
        return data_term + (theta.mT @ penalty @ theta).reshape(theta.shape[0])

    eta = offset.clone()
    theta = None
    for _ in range(max_iter):
        working_response, weights = strategy.working_response_and_weights(y, eta)
        weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(_TINY)
        cov_x, cov_xy = _compute_weighted_covariances(phi, working_response - offset, loss, weights)
        candidate = solve_linear_system(cov_x, cov_xy, penalty, num_samples)
        eta_candidate = phi @ candidate + offset
        if theta is not None:
            full_step = candidate
            previous = objective(theta, eta)
            current = objective(full_step, eta_candidate)
            step = torch.ones(theta.shape[0], 1, 1, dtype=dtype, device=phi.device)
            for _ in range(max_halvings):
                worse = ~torch.isfinite(current) | (current > previous)
                if not bool(worse.any()):
                    break
                step = torch.where(worse.view(-1, 1, 1), step * 0.5, step)
                candidate = theta + step * (full_step - theta)
                eta_candidate = phi @ candidate + offset
                current = objective(candidate, eta_candidate)
        change = (eta_candidate - eta).flatten(1).norm(dim=1) / eta.flatten(1).norm(dim=1).clamp_min(_TINY)
        theta, eta = candidate, eta_candidate
        if bool((change < tol).all()):
            break
    return theta
