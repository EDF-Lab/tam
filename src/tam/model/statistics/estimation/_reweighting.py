# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The iteratively reweighted penalized least-squares loop.

Turns the single P-WLS atom (BaseTAM._solve_pwls_step) into a fit for any strategy.
Each pass asks the strategy for (z, w), normalises w to unit mean so the n*S penalty
scale is preserved, and solves one system. GLM families get step-halving on the
penalized objective to stay monotone. The Gaussian strategy converges in one step and
reproduces ordinary least squares exactly.

The design matrix does not depend on the coefficients, the working response or the weights: when it fits in memory it is
built once per fit (``dense_design``) and every pass, every step-halving and every dispersion refit reuses it. Otherwise the
loop falls back on the chunked solver of the dispatcher, which builds it block by block.
"""
from __future__ import annotations

import torch

from tam.common.hardware import hw
from tam.model._math import _compute_weighted_covariances, solve_linear_system
from tam.model._memory import can_fit_dense_matrix
from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-12
#: Switch of the design-matrix cache (tests compare the cached loop with the rebuilding one).
_DENSE_CACHE: bool = True
#: A cached design matrix, its weighted copy and the covariance workspace stay under this share of the free memory.
_CACHE_MEMORY_SHARE: float = 0.4
_CACHE_COPIES: int = 3


def dense_design(model, x_data: torch.Tensor):
    """The design matrix Phi(x), built once, or None when it does not fit in memory (the loop then rebuilds it by chunks)."""
    if not _DENSE_CACHE:
        return None
    probe = model._build_design_matrix(x_data[:, 0:1, :])
    total_d = int(probe.shape[-1])
    if not can_fit_dense_matrix(total_d, x_data.device, batch_size=1):
        return None
    n_groups, n_samples = int(x_data.shape[0]), int(x_data.shape[1])
    if n_groups * n_samples * total_d * 8 * _CACHE_COPIES > hw.get_available_memory() * _CACHE_MEMORY_SHARE:
        return None
    try:
        return model._build_design_matrix(x_data)
    except (torch.OutOfMemoryError, MemoryError):
        return None


def _linear_predictor(model, x_data: torch.Tensor, theta: torch.Tensor, design_matrix=None) -> torch.Tensor:
    """eta = Phi(x) @ theta on the model's design matrix (the cached one when given)."""
    if design_matrix is None:
        design_matrix = model._build_design_matrix(x_data)
    eta = design_matrix @ theta.to(device=design_matrix.device, dtype=design_matrix.dtype)
    offset = model._offset_of(x_data)
    return eta if offset is None else eta + offset.to(device=eta.device, dtype=eta.dtype)


def _solve_step(model, x_data, working_response, weights, design_matrix, penalty_matrix, loss_matrix):
    """One penalized weighted least-squares solve; on the cached design matrix when there is one."""
    if design_matrix is None:
        return model._solve_pwls_step(x_data, working_response, weights=weights)
    offset = model._offset_of(x_data)
    if offset is not None:
        working_response = working_response - offset.to(device=working_response.device, dtype=working_response.dtype)
    cov_x, cov_xy = _compute_weighted_covariances(design_matrix, working_response, loss_matrix, weights)
    return solve_linear_system(cov_x, cov_xy, penalty_matrix, x_data.shape[1]).cpu()


#: <reweighting_loop>
def reweighted_penalized_fit(
    model,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    strategy: ReweightingStrategy,
    max_iter: int = 25,
    tol: float = 1e-6,
    max_halvings: int = 8,
    design_matrix=None,
) -> torch.Tensor:
    """Fit a strategy by iteratively reweighted penalized least squares; return the coefficients.

    Args:
        model: a TAM exposing _solve_pwls_step and _build_design_matrix.
        x_data: feature tensor (n_groups, n_samples, n_features).
        y_data: response tensor (n_groups, n_samples, 1).
        strategy: the reweighting rule producing (z, w) per iteration.
        max_iter: maximum outer iterations.
        tol: relative change in eta for convergence.
        max_halvings: maximum step-halvings per iteration (GLM divergence guard).
        design_matrix: Phi(x) when the caller already built it (``dense_design``); built here when None.
    """
    y_data = y_data.to(torch.get_default_dtype())
    eta = strategy.initial_eta(y_data)
    theta = None

    penalty_matrix = model._build_penalty_matrix()
    if design_matrix is None:
        design_matrix = dense_design(model, x_data)
    loss_matrix = model._build_loss_matrix() if design_matrix is not None else None

    def penalized_objective(theta_value: torch.Tensor, eta_value: torch.Tensor) -> torch.Tensor:
        # mean deviance + roughness penalty theta.T * S * theta. Step-halving must judge the
        # penalized objective, else a step that lowers deviance but inflates roughness is accepted.
        base = strategy.mean_objective(y_data, eta_value)
        penalty_quadratic = theta_value.mT @ penalty_matrix.to(theta_value.dtype) @ theta_value
        return base + penalty_quadratic.mean()

    for _ in range(max_iter):
        working_response, weights = strategy.working_response_and_weights(y_data, eta)
        weights = weights / weights.mean().clamp_min(_TINY)
        theta_candidate = _solve_step(model, x_data, working_response, weights, design_matrix, penalty_matrix, loss_matrix)
        eta_candidate = _linear_predictor(model, x_data, theta_candidate, design_matrix)

        if theta is not None and strategy.is_glm:
            theta_full_step = theta_candidate
            objective_previous = penalized_objective(theta, eta)
            objective_candidate = penalized_objective(theta_candidate, eta_candidate)
            step = 1.0
            halving = 0
            while (not torch.isfinite(objective_candidate) or objective_candidate > objective_previous) \
                    and halving < max_halvings:
                step *= 0.5
                theta_candidate = theta + step * (theta_full_step - theta)
                eta_candidate = _linear_predictor(model, x_data, theta_candidate, design_matrix)
                objective_candidate = penalized_objective(theta_candidate, eta_candidate)
                halving += 1

        relative_change = (eta_candidate - eta).norm() / eta.norm().clamp_min(_TINY)
        theta = theta_candidate
        eta = eta_candidate
        if relative_change < tol:
            break

    return theta
#: </reweighting_loop>
