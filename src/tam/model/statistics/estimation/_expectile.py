# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Expectiles: the asymmetric-least-squares strategy and the expectile<->quantile map.

ExpectileLoss weights residuals by tau when under-predicted, else 1-tau (z = y).
empirical_expectile / expectile_level_for_quantile give the Jones/Yao-Tong bijection
so an expectile fit can be read back as a genuine quantile.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import optimize

from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-9


class ExpectileLoss(ReweightingStrategy):
    """Asymmetric least squares (Newey-Powell): weight tau above the fit, else 1-tau."""

    name = "expectile"

    def __init__(self, tau: float = 0.5):
        if not 0.0 < tau < 1.0:
            raise ValueError(f"expectile level tau must be in (0, 1); got {tau}")
        self.tau = float(tau)

    def working_response_and_weights(self, y, eta):
        residual = y - eta
        weights = torch.where(residual >= 0, torch.as_tensor(self.tau, dtype=y.dtype, device=y.device),
                              torch.as_tensor(1.0 - self.tau, dtype=y.dtype, device=y.device))
        return y, weights

    def mean_objective(self, y, eta):
        residual = y - eta
        asymmetric = torch.where(residual >= 0, self.tau, 1.0 - self.tau)
        return (asymmetric * residual * residual).mean()


def empirical_expectile(sample: np.ndarray, tau: float) -> float:
    """The tau-expectile of a sample (asymmetric first-moment condition)."""
    values = np.asarray(sample, dtype=float)
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got {tau}")
    lower_bound, upper_bound = float(values.min()), float(values.max())
    if lower_bound == upper_bound:
        return lower_bound

    def estimating_equation(mu: float) -> float:
        upper_partial = np.clip(values - mu, 0.0, None).sum()
        lower_partial = np.clip(mu - values, 0.0, None).sum()
        return tau * upper_partial - (1.0 - tau) * lower_partial

    return float(optimize.brentq(estimating_equation, lower_bound, upper_bound))


def expectile_level_for_quantile(sample: np.ndarray, alpha: float) -> float:
    """The expectile level h(alpha) whose value equals the alpha-quantile of the sample.

    Jones/Yao-Tong identity: h = (G - alpha*q) / (2*G - m + (1 - 2*alpha)*q), with q the
    alpha-quantile, m the mean and G the lower partial first moment E[Y * 1{Y <= q}].
    """
    values = np.asarray(sample, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    quantile = float(np.quantile(values, alpha))
    mean = float(values.mean())
    partial_first_moment = float(values[values <= quantile].sum() / values.size)
    numerator = partial_first_moment - alpha * quantile
    denominator = 2.0 * partial_first_moment - mean + (1.0 - 2.0 * alpha) * quantile
    if abs(denominator) < _TINY:
        return alpha
    return float(np.clip(numerator / denominator, _TINY, 1.0 - _TINY))
