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

from typing import Optional
import torch

from tam.common.hardware import hw
from tam.model._math import _StructuredDesign, _compute_weighted_covariances, solve_linear_system
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
    effects_list = getattr(model, "effects_list_", None)
    if effects_list is not None:
        total_d = sum(e.get_n_coeffs() for e in effects_list)
    else:
        probe = model._build_design_matrix(x_data[:, 0:1, :])
        total_d = int(probe.shape[-1])
    if not can_fit_dense_matrix(total_d, x_data.device, batch_size=1):
        return None
    n_groups, n_samples = int(x_data.shape[0]), int(x_data.shape[1])
    if n_groups * n_samples * total_d * 8 * _CACHE_COPIES > hw.get_available_memory() * _CACHE_MEMORY_SHARE:
        return None
    loss_matrix = model._build_loss_matrix() if hasattr(model, "_build_loss_matrix") else None
    if effects_list is not None and (loss_matrix is None or loss_matrix.shape == (1, 1)):
        feat_cols = (getattr(model, "features_config_", None) or {}).get("features")
        try:
            struct = _StructuredDesign.try_build(x_data, effects_list, feature_columns=feat_cols)
        except (torch.OutOfMemoryError, MemoryError):
            struct = None
        if struct is not None:
            holder = torch.empty((0,), device=struct.device, dtype=struct.dtype)
            holder._tam_structured = struct
            return holder
    try:
        return model._build_design_matrix(x_data)
    except (torch.OutOfMemoryError, MemoryError):
        return None


def _linear_predictor(model, x_data: torch.Tensor, theta: torch.Tensor, design_matrix=None) -> torch.Tensor:
    """eta = Phi(x) @ theta on the model's design matrix (the cached one when given)."""
    struct = getattr(design_matrix, "_tam_structured", None) if design_matrix is not None else None
    if struct is not None:
        eta = struct.linear_predictor(theta)
    else:
        if design_matrix is None:
            design_matrix = model._build_design_matrix(x_data)
        eta = design_matrix @ theta.to(device=design_matrix.device, dtype=design_matrix.dtype)
    offset = model._offset_of(x_data)
    if offset is not None:
        eta = eta + offset.to(device=eta.device, dtype=eta.dtype)
    return eta.to(x_data.device)


def _solve_step(model, x_data, working_response, weights, design_matrix, penalty_matrix, loss_matrix):
    """One penalized weighted least-squares solve; on the cached design matrix when there is one."""
    if design_matrix is None:
        return model._solve_pwls_step(x_data, working_response, weights=weights).to(x_data.device)
    offset = model._offset_of(x_data)
    if offset is not None:
        working_response = working_response - offset.to(device=working_response.device, dtype=working_response.dtype)
    struct = getattr(design_matrix, "_tam_structured", None)
    if struct is not None and loss_matrix.shape == (1, 1):
        cov_x, cov_xy = struct.weighted_covariances(working_response, loss_matrix, weights)
    else:
        cov_x, cov_xy = _compute_weighted_covariances(design_matrix, working_response, loss_matrix, weights)
    return solve_linear_system(cov_x, cov_xy, penalty_matrix, x_data.shape[1]).to(x_data.device)


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
    initial_theta: Optional[torch.Tensor] = None,
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
        initial_theta: optional warm-start coefficients from a preceding outer fit.
    """
    target_device = y_data.device
    target_dtype = torch.get_default_dtype()
    y_data = y_data.to(device=target_device, dtype=target_dtype)

    if design_matrix is None:
        design_matrix = dense_design(model, x_data)
    penalty_matrix = model._build_penalty_matrix().to(device=target_device, dtype=target_dtype)
    loss_matrix = model._build_loss_matrix().to(device=target_device) if design_matrix is not None else None

    # A cold start keeps the plain iteration (the same numbers as before); the Anderson window only speeds up the warm-started passes of an outer search.
    accelerate = initial_theta is not None
    if initial_theta is not None:
        theta = initial_theta.to(device=target_device, dtype=target_dtype)
        eta = _linear_predictor(model, x_data, theta, design_matrix).to(device=target_device, dtype=target_dtype)
    else:
        theta = None
        eta = strategy.initial_eta(y_data).to(device=target_device, dtype=target_dtype)

    def penalized_objective(theta_value: torch.Tensor, eta_value: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        # mean deviance / w_mean + roughness penalty theta.T * S * theta, matching the unit-mean-normalized working system
        # (Phi' (W / w_mean) Phi + n S) theta = Phi' (W / w_mean) z solved by _solve_step.
        eta_aligned = eta_value.to(device=target_device, dtype=target_dtype)
        theta_aligned = theta_value.to(device=target_device, dtype=target_dtype)
        base = strategy.mean_objective(y_data, eta_aligned) / w_scale
        penalty_quadratic = theta_aligned.mT @ penalty_matrix @ theta_aligned
        return base + penalty_quadratic.mean()

    aa_g_hist = []
    aa_f_hist = []
    aa_max_m = 3

    for iter_idx in range(max_iter):
        working_response, weights = strategy.working_response_and_weights(y_data, eta)
        w_mean = weights.mean().clamp_min(_TINY)
        weights = weights / w_mean
        theta_candidate = _solve_step(
            model, x_data, working_response, weights, design_matrix, penalty_matrix, loss_matrix
        ).to(device=target_device, dtype=target_dtype)
        eta_candidate = _linear_predictor(model, x_data, theta_candidate, design_matrix).to(
            device=target_device, dtype=target_dtype
        )

        if iter_idx >= 1 and theta is not None and strategy.is_glm:
            theta_full_step = theta_candidate
            objective_previous = penalized_objective(theta, eta, w_mean)
            objective_candidate = penalized_objective(theta_candidate, eta_candidate, w_mean)
            step = 1.0
            halving = 0
            while (not torch.isfinite(objective_candidate) or objective_candidate > objective_previous) \
                    and halving < max_halvings:
                step *= 0.5
                theta_candidate = theta + step * (theta_full_step - theta)
                eta_candidate = _linear_predictor(model, x_data, theta_candidate, design_matrix).to(
                    device=target_device, dtype=target_dtype
                )
                objective_candidate = penalized_objective(theta_candidate, eta_candidate, w_mean)
                halving += 1
            if halving == max_halvings and (not torch.isfinite(objective_candidate) or objective_candidate > objective_previous):
                break

            # Safeguarded Type-II Anderson acceleration (window m = 3) to accelerate fixed-point PIRLS contraction.
            g_flat = theta_candidate.reshape(-1)
            f_flat = g_flat - theta.reshape(-1)
            aa_g_hist.append(g_flat)
            aa_f_hist.append(f_flat)
            if len(aa_g_hist) > aa_max_m + 1:
                aa_g_hist.pop(0)
                aa_f_hist.pop(0)
            if accelerate and iter_idx >= 2 and len(aa_f_hist) >= 3 and torch.isfinite(objective_candidate):
                k_aa = len(aa_f_hist) - 1
                d_g = torch.stack([aa_g_hist[j + 1] - aa_g_hist[j] for j in range(k_aa)], dim=1)
                d_f = torch.stack([aa_f_hist[j + 1] - aa_f_hist[j] for j in range(k_aa)], dim=1)
                gram = d_f.T @ d_f
                scale_diag = float(gram.diagonal().abs().max().item())
                if scale_diag > 1e-24:
                    gram.diagonal().add_(1e-10 * scale_diag)
                    try:
                        gamma_aa = torch.linalg.solve(gram, d_f.T @ f_flat)
                        if float(gamma_aa.abs().max().item()) <= 50.0:
                            theta_aa = (g_flat - d_g @ gamma_aa).view_as(theta_candidate)
                            eta_aa = _linear_predictor(model, x_data, theta_aa, design_matrix).to(
                                device=target_device, dtype=target_dtype
                            )
                            obj_aa = penalized_objective(theta_aa, eta_aa, w_mean)
                            if torch.isfinite(obj_aa) and obj_aa < objective_candidate:
                                theta_candidate = theta_aa
                                eta_candidate = eta_aa
                    except RuntimeError:
                        pass

        diff_eta = eta_candidate - eta
        relative_change = float((diff_eta.norm() / eta.norm().clamp_min(_TINY)).item())
        if iter_idx >= 1 and strategy.is_glm:
            root_w = weights.clamp_min(0.0).sqrt()
            w_rel = float(((diff_eta * root_w).norm() / (eta * root_w).norm().clamp_min(_TINY)).item())
            relative_change = min(relative_change, w_rel)
        theta = theta_candidate
        eta = eta_candidate
        if relative_change < tol:
            break

    return theta.cpu()
#: </reweighting_loop>
