# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Adaptive Conformal Inference (ACI): the streaming, non-stationary risk-control loop.

Split conformal (SafetyTAM) guarantees coverage only under exchangeability, which drift breaks. ACI
(Gibbs & Candes 2021) replaces the fixed miscoverage level with an online update on alpha_t:

    alpha_{t+1} = alpha_t + gamma * (alpha_target - error_t),   error_t = 1{Y_t not in C_t}

A miscoverage lowers alpha (a wider next interval); a covered step raises it. This module holds only that
control loop and drives any calibrated radius provider: a callable radius_fn(alpha) -> radius, in practice
a calibrated SafetyTAM.conformal_quantile. It never re-implements the finite-sample quantile.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

# The risk level is clamped to this open interval so the effective quantile can never degenerate.
_ALPHA_FLOOR: float = 1e-3
_ALPHA_CEIL: float = 0.999


def update_risk_level(alpha_t: float, error: float, alpha_target: float, gamma: float) -> float:
    """One ACI integrator step: alpha_{t+1} = alpha_t + gamma * (alpha_target - error).

    error is the realized miscoverage indicator (1.0 if the point fell outside the interval, else 0.0).
    This is the raw, unbounded integrator of Gibbs & Candes: the level is intentionally not clamped here so
    cumulative miscoverage under sustained drift accumulates. It is clamped only where used - at the quantile
    call, via effective_alpha.
    """
    return float(alpha_t + gamma * (alpha_target - error))


def effective_alpha(alpha_t: float) -> float:
    """Clamp the (possibly unbounded) integrator level to the safe open interval used for the quantile."""
    return float(np.clip(alpha_t, _ALPHA_FLOOR, _ALPHA_CEIL))


def adaptive_conformal_scores(
    radius_fn: Callable[[float], float],
    score_stream: np.ndarray,
    alpha_target: float,
    gamma: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ACI over a stream of realized nonconformity scores.

    radius_fn(alpha) -> radius is a calibrated conformal quantile. Returns (radii, alphas): the conformal
    radius used and the adapted risk level at each step. A point is covered iff its score is within the radius.
    """
    stream = np.atleast_1d(np.asarray(score_stream, dtype=float))
    radii = np.empty(len(stream))
    alphas = np.empty(len(stream))
    current_alpha = alpha_target
    for t, score in enumerate(stream):
        alpha_eff = effective_alpha(current_alpha)
        radius = radius_fn(alpha_eff)
        radii[t] = radius
        alphas[t] = alpha_eff
        error = 0.0 if score <= radius else 1.0
        current_alpha = update_risk_level(current_alpha, error, alpha_target, gamma)
    return radii, alphas


def adaptive_conformal_intervals(
    radius_fn: Callable[[float], float],
    y_pred: np.ndarray,
    y_true_online: np.ndarray,
    alpha_target: float,
    gamma: float = 0.05,
    scale: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Streaming ACI prediction intervals [y_pred +/- q_t * scale] whose level adapts online.

    radius_fn(alpha) -> radius is a calibrated conformal quantile; y_true_online (required) is the realized
    values observed sequentially. Returns a DataFrame with
    ['Predicted', 'Lower', 'Upper', 'Alpha_t', 'Width', 'Actual', 'Covered'].
    """
    if y_true_online is None:
        raise ValueError("ACI requires 'y_true_online' to update its risk level.")
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_online = np.asarray(y_true_online, dtype=float)
    n = len(y_pred)
    scale_vector = np.ones(n) if scale is None else np.asarray(scale, dtype=float)
    radii = np.zeros(n)
    alphas = np.zeros(n)
    current_alpha = alpha_target
    for t in range(n):
        alpha_eff = effective_alpha(current_alpha)
        radius = radius_fn(alpha_eff)
        radii[t] = radius
        alphas[t] = alpha_eff
        half_width = radius * scale_vector[t]
        is_covered = (y_pred[t] - half_width <= y_true_online[t] <= y_pred[t] + half_width)
        current_alpha = update_risk_level(current_alpha, 0.0 if is_covered else 1.0, alpha_target, gamma)

    half_widths = radii * scale_vector
    result = pd.DataFrame({
        'Predicted': y_pred,
        'Lower': y_pred - half_widths,
        'Upper': y_pred + half_widths,
        'Alpha_t': alphas,
        'Width': 2 * half_widths,
    })
    result['Actual'] = y_true_online
    result['Covered'] = (result['Actual'] >= result['Lower']) & (result['Actual'] <= result['Upper'])
    return result
