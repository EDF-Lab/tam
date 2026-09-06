# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Epistemic (parameter) uncertainty via the Bayesian interpretation of the penalized fit.

The penalized least-squares solution is the MAP estimate under a Gaussian prior induced by the roughness
penalty S; the posterior covariance is V_beta = sigma^2 (Phi.T Phi + n S)^-1. The predictive epistemic
standard error at a point is sqrt(phi V_beta phi.T) - it widens where the training design is data-poor.
This complements the aleatoric scale sigma(x) of a distributional StaticTAM. State is not stored on the
model (recomputed on demand); single-group only.
"""
from __future__ import annotations

import pandas as pd
import torch
from scipy import stats

from tam.common.utils import _ensure_dummies, _balance_groups


def _prepared_tensors(model, data: pd.DataFrame, with_target: bool):
    dummied = _ensure_dummies(data.copy(), model.group_col_, model.date_col_)
    method = "drop" if with_target else "fill"
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method=method)
    x_tensor, y_tensor, _ = model._prepare_data(
        balanced, target_col=model.target_col_ if with_target else None
    )
    return x_tensor, y_tensor


def posterior_prediction(
    model,
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    level: float = 0.95,
) -> pd.DataFrame:
    """Epistemic (parameter-uncertainty) prediction interval for a fitted single-group StaticTAM.

    Returns a frame indexed like new_data with the linear-predictor prediction, its epistemic standard error
    epistemic_std, and the `level` credible band (lower, upper).

    Note: this models epistemic (parameter) uncertainty only, with the noise as a single global scalar
    sigma^2. On heteroscedastic data (what a distributional StaticTAM is built for) this global variance
    under-covers in high-variance regions and over-covers in low-variance ones; pair it with a distributional
    StaticTAM's sigma(x) for the aleatoric component. The posterior is solved through a Cholesky factor of
    (Phi.T Phi + n S) - stabler than an explicit inverse, and it never materialises the dense inverse or the
    full posterior covariance (only triangular solves against Phi_new.T).
    """
    if model.coefficients_ is None:
        raise RuntimeError("Model must be fitted before computing posterior uncertainty.")

    x_train, y_train = _prepared_tensors(model, train_data, with_target=True)
    if x_train.shape[0] != 1:
        raise NotImplementedError("posterior_prediction currently supports single-group models only.")

    design_train = model._build_design_matrix(x_train.to(torch.float64))
    penalty = model._build_penalty_matrix().to(torch.float64)
    coefficients = model.coefficients_.to(torch.float64)
    n_samples = x_train.shape[1]
    n_coeffs = design_train.shape[-1]

    gram = design_train.mT @ design_train
    regularized = gram + n_samples * penalty + 1e-6 * n_samples * torch.eye(n_coeffs, dtype=torch.float64)

    # Factor (Phi.T Phi + nS) once; reuse via triangular solves so neither the dense inverse nor the full
    # posterior covariance is ever materialised. cholesky_ex is non-throwing; on a rare non-PD factorisation
    # add a stronger ridge and retry.
    cholesky_factor, info = torch.linalg.cholesky_ex(regularized)
    if bool((info != 0).any()):
        regularized = regularized + 1e-3 * n_samples * torch.eye(n_coeffs, dtype=torch.float64)
        cholesky_factor, _ = torch.linalg.cholesky_ex(regularized)

    residual = y_train.to(torch.float64) - design_train @ coefficients
    residual_sum_of_squares = float((residual ** 2).sum())
    # Effective dof = tr((Phi.T Phi + nS)^-1 Phi.T Phi), via a Cholesky solve (no explicit inverse).
    effective_dof = float(torch.einsum("gii->g", torch.cholesky_solve(gram, cholesky_factor)).sum())
    noise_variance = residual_sum_of_squares / max(n_samples - effective_dof, 1.0)

    x_new, _ = _prepared_tensors(model, new_data, with_target=False)
    design_new = model._build_design_matrix(x_new.to(torch.float64))
    prediction = (design_new @ coefficients).squeeze(-1).squeeze(0).cpu().numpy()
    # diag(Phi_new (Phi.T Phi + nS)^-1 Phi_new.T) via a solve against Phi_new.T (no D x D covariance).
    covariance_solved = torch.cholesky_solve(design_new.mT, cholesky_factor)
    prediction_variance = (noise_variance * torch.einsum(
        "gnd,gdn->gn", design_new, covariance_solved
    )).clamp_min(0.0)
    epistemic_std = prediction_variance.sqrt().squeeze(0).cpu().numpy()

    z_value = float(stats.norm.ppf(0.5 + level / 2.0))
    return pd.DataFrame({
        "prediction": prediction,
        "epistemic_std": epistemic_std,
        "lower": prediction - z_value * epistemic_std,
        "upper": prediction + z_value * epistemic_std,
    }, index=new_data.index)
