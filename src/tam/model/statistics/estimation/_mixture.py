# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Gaussian-mixture-of-TAM-regressions schedule and EM numerics.

A mixture is a schedule over the single P-WLS atom, not a new solver: the M-step is the weighted
atom (_solve_pwls_step) with the posterior responsibilities as sample_weights; the E-step recomputes
those from the current component means and scales. StaticTAM stays a thin frontend and delegates here.
Single-group only (one cross-section / one time series), matching the atom's per-group batching.

This module is a self-contained sibling of _distributional.py: it duplicates the trivial log-target helper on
purpose so the two distributional extensions stay fully decoupled.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from tam.common.utils import _ensure_dummies, _balance_groups
from ..._base import BaseTAM

_TINY: float = 1e-12


def _to_model_scale(model, y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y, _TINY, None)) if model._log_target_ else np.asarray(y, dtype=float)


def require_mixture(model, method_name: str) -> None:
    if getattr(model, "_mixture_components_", None) is None:
        raise RuntimeError(
            f"{method_name}() is only available on a mixture StaticTAM (mixture_components=K)."
        )


# --- EM numerics ------------------------------------------------------------------------------------
def _single_em_run(model, x_tensor, y_tensor, design_matrix, target, n_components, max_iter, tol, seed) -> dict:
    """One EM run from a seeded responsibility initialisation; returns the fitted params and log-likelihood."""
    n_samples = target.shape[0]
    generator = np.random.default_rng(seed)
    component_of_rank = np.floor(np.argsort(np.argsort(target)) / n_samples * n_components).astype(int)
    responsibilities = np.full((n_samples, n_components), 0.05 / n_components)
    responsibilities[np.arange(n_samples), np.clip(component_of_rank, 0, n_components - 1)] += 0.95
    responsibilities += generator.uniform(0, 1e-3, responsibilities.shape)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    coefficients: list = []
    scales = np.empty(n_components)
    mixing_weights = np.ones(n_components) / n_components
    previous_log_likelihood = -np.inf
    for _ in range(max_iter):
        coefficients = []
        scales = np.empty(n_components)
        weights = np.empty(n_components)
        for component in range(n_components):
            responsibility = responsibilities[:, component]
            weight_tensor = torch.as_tensor(
                responsibility / max(responsibility.mean(), _TINY), dtype=torch.float64
            ).reshape(1, n_samples, 1)
            theta = model._solve_pwls_step(x_tensor, y_tensor, weights=weight_tensor)
            mean = (design_matrix @ theta).squeeze(-1).squeeze(0).cpu().numpy()
            residual = target - mean
            variance = float(np.sum(responsibility * residual ** 2) / max(responsibility.sum(), _TINY))
            coefficients.append(theta)
            scales[component] = np.sqrt(max(variance, _TINY))
            weights[component] = responsibility.mean()
        mixing_weights = weights / weights.sum()

        means = np.stack(
            [(design_matrix @ theta).squeeze(-1).squeeze(0).cpu().numpy() for theta in coefficients], axis=1
        )
        component_density = mixing_weights * norm.pdf(target[:, None], loc=means, scale=scales[None, :])
        total_density = component_density.sum(axis=1)
        responsibilities = component_density / np.clip(total_density[:, None], _TINY, None)
        log_likelihood = float(np.sum(np.log(np.clip(total_density, _TINY, None))))
        if abs(log_likelihood - previous_log_likelihood) < tol * (abs(previous_log_likelihood) + 1.0):
            previous_log_likelihood = log_likelihood
            break
        previous_log_likelihood = log_likelihood
    return {
        "coefficients": coefficients,
        "scales": scales,
        "weights": mixing_weights,
        "log_likelihood": previous_log_likelihood,
    }


def fit_mixture_em(model, x_tensor, y_tensor, design_matrix, target, n_components,
                   max_iter: int = 100, tol: float = 1e-5, seed: int = 0, n_init: int = 1) -> dict:
    """Fit a K-component Gaussian mixture by EM, restarting n_init times and keeping the best fit."""
    best_fit = None
    for restart in range(max(int(n_init), 1)):
        candidate = _single_em_run(
            model, x_tensor, y_tensor, design_matrix, target, n_components, max_iter, tol, seed + restart
        )
        if best_fit is None or candidate["log_likelihood"] > best_fit["log_likelihood"]:
            best_fit = candidate
    return best_fit


# --- Orchestration (thin StaticTAM delegates land here) ---------------------------------------------
def _prepare_for_em(model, data, with_target: bool):
    """Balance + tensorise like the fit path, returning (x_tensor, y_tensor) for the EM loop."""
    dummied = _ensure_dummies(data.copy(), model.group_col_, model.date_col_)
    method = "drop" if with_target else "fill"
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method=method)
    x_tensor, y_tensor, _ = model._prepare_data(
        balanced, target_col=model.target_col_ if with_target else None
    )
    return x_tensor, y_tensor


def fit(model, data):
    """Fit a K-component Gaussian mixture by EM; the M-step is the responsibility-weighted atom."""
    working = data.copy()
    working[model.target_col_] = _to_model_scale(model, working[model.target_col_].to_numpy())

    # Establish normalization, effects and a baseline L2 fit on the (log-)target scale.
    BaseTAM.fit(model, working)

    x_tensor, y_tensor = _prepare_for_em(model, working, with_target=True)
    if x_tensor.shape[0] != 1:
        raise NotImplementedError("Mixture StaticTAM currently supports single-group models only.")
    design_matrix = model._build_design_matrix(x_tensor.to(torch.float64))
    target = y_tensor.to(torch.float64).squeeze(-1).squeeze(0).cpu().numpy()

    best_fit = fit_mixture_em(
        model, x_tensor, y_tensor, design_matrix, target, model._mixture_components_,
        max_iter=model._mixture_max_iter_, tol=model._mixture_tol_,
        seed=model._mixture_seed_, n_init=model._mixture_n_init_,
    )
    model.component_coefficients_ = best_fit["coefficients"]
    model.component_scales_ = best_fit["scales"]
    model.mixing_weights_ = best_fit["weights"]
    model.log_likelihood_ = best_fit["log_likelihood"]
    return model


def _mixture_design(model, data):
    x_tensor, _ = _prepare_for_em(model, data, with_target=False)
    return model._build_design_matrix(x_tensor.to(torch.float64))


def _component_means_matrix(model, design_matrix):
    means = [(design_matrix @ theta).squeeze(-1).squeeze(0).cpu().numpy()
             for theta in model.component_coefficients_]
    return np.stack(means, axis=1)


def component_means(model, data):
    """Per-component conditional means on the response scale. Shape (n, K)."""
    require_mixture(model, "component_means")
    means = _component_means_matrix(model, _mixture_design(model, data))
    return np.exp(means) if model._log_target_ else means


def predict_mean(model, data):
    """The mixture mean on the response scale."""
    require_mixture(model, "predict_mean")
    means = _component_means_matrix(model, _mixture_design(model, data))
    if model._log_target_:
        component_response_mean = np.exp(means + 0.5 * model.component_scales_[None, :] ** 2)
    else:
        component_response_mean = means
    return np.sum(model.mixing_weights_[None, :] * component_response_mean, axis=1)


def responsibilities(model, data):
    """Posterior component membership at each observed target. Shape (n, K)."""
    require_mixture(model, "responsibilities")
    means = _component_means_matrix(model, _mixture_design(model, data))
    target = _to_model_scale(model, data[model.target_col_].to_numpy())
    density = model.mixing_weights_ * norm.pdf(target[:, None], loc=means, scale=model.component_scales_[None, :])
    return density / np.clip(density.sum(axis=1, keepdims=True), _TINY, None)


def log_density(model, data):
    """Mixture log predictive density at each observed target (model scale)."""
    require_mixture(model, "log_density")
    means = _component_means_matrix(model, _mixture_design(model, data))
    target = _to_model_scale(model, data[model.target_col_].to_numpy())
    density = model.mixing_weights_ * norm.pdf(target[:, None], loc=means, scale=model.component_scales_[None, :])
    return np.log(np.clip(density.sum(axis=1), _TINY, None))


def anomaly_score(model, data):
    """Negative mixture log-density; large where the observation is improbable under every component."""
    return -log_density(model, data)
