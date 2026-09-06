# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
The iteratively reweighted penalized least-squares loop.

Turns the single P-WLS atom (BaseTAM._solve_pwls_step) into a fit for any strategy.
Each pass asks the strategy for (z, w), normalises w to unit mean so the n*S penalty
scale is preserved, and solves one system. GLM families get step-halving on the
penalized objective to stay monotone. The Gaussian strategy converges in one step and
reproduces ordinary least squares exactly.
"""
from __future__ import annotations

import torch

from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-12


def _linear_predictor(model, x_data: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """eta = Phi(x) @ theta on the model's design matrix."""
    design_matrix = model._build_design_matrix(x_data)
    return design_matrix @ theta.to(design_matrix.dtype)


#: <reweighting_loop>
def reweighted_penalized_fit(
    model,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    strategy: ReweightingStrategy,
    max_iter: int = 25,
    tol: float = 1e-6,
    max_halvings: int = 8,
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
    """
    y_data = y_data.to(torch.get_default_dtype())
    eta = strategy.initial_eta(y_data)
    theta = None

    penalty_matrix = model._build_penalty_matrix()

    def penalized_objective(theta_value: torch.Tensor, eta_value: torch.Tensor) -> torch.Tensor:
        # mean deviance + roughness penalty theta.T * S * theta. Step-halving must judge the
        # penalized objective, else a step that lowers deviance but inflates roughness is accepted.
        base = strategy.mean_objective(y_data, eta_value)
        penalty_quadratic = theta_value.mT @ penalty_matrix.to(theta_value.dtype) @ theta_value
        return base + penalty_quadratic.mean()

    for _ in range(max_iter):
        working_response, weights = strategy.working_response_and_weights(y_data, eta)
        weights = weights / weights.mean().clamp_min(_TINY)
        theta_candidate = model._solve_pwls_step(x_data, working_response, weights=weights)
        eta_candidate = _linear_predictor(model, x_data, theta_candidate)

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
                eta_candidate = _linear_predictor(model, x_data, theta_candidate)
                objective_candidate = penalized_objective(theta_candidate, eta_candidate)
                halving += 1

        relative_change = (eta_candidate - eta).norm() / eta.norm().clamp_min(_TINY)
        theta = theta_candidate
        eta = eta_candidate
        if relative_change < tol:
            break

    return theta
#: </reweighting_loop>
