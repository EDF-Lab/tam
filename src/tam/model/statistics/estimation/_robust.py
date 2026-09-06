# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Robust M-estimators (Huber, Student-t).

Not GLMs: the working response is just z = y, and the loss enters through the weights
w = psi(u)/u, computed against a MAD-based robust scale. Weights are always positive.
"""
from __future__ import annotations

import torch

from ._base_strategy import ReweightingStrategy

_TINY: float = 1e-12


def _robust_scale(residual: torch.Tensor) -> torch.Tensor:
    """Robust scale from the median absolute deviation (consistent for the normal)."""
    median = residual.median()
    mad = (residual - median).abs().median()
    return (1.4826 * mad).clamp_min(_TINY)


class HuberLoss(ReweightingStrategy):
    """Huber: weight 1 within delta robust scales, else delta/|u| (bounded influence)."""

    name = "huber"

    def __init__(self, delta: float = 1.345):
        if delta <= 0:
            raise ValueError(f"huber delta must be positive; got {delta}")
        self.delta = float(delta)

    def working_response_and_weights(self, y, eta):
        residual = y - eta
        scale = _robust_scale(residual)
        standardized = (residual / scale).abs()
        weights = torch.where(standardized <= self.delta, torch.ones_like(standardized),
                              self.delta / standardized.clamp_min(_TINY))
        return y, weights


class StudentTLoss(ReweightingStrategy):
    """Student-t (dof nu): weight (nu + 1) / (nu + (u/scale)^2) down-weights outliers."""

    name = "student_t"

    def __init__(self, nu: float = 4.0):
        if nu <= 0:
            raise ValueError(f"student_t nu (degrees of freedom) must be positive; got {nu}")
        self.nu = float(nu)

    def working_response_and_weights(self, y, eta):
        residual = y - eta
        scale = _robust_scale(residual)
        standardized_sq = (residual / scale) ** 2
        weights = (self.nu + 1.0) / (self.nu + standardized_sq)
        return y, weights
