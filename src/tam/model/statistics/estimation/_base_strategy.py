# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
The reweighting-strategy contract (mirror of spectrum/_base_effects.py).

A strategy is the rule that turns the current linear predictor eta into the
working response z and per-observation weights w for the next P-WLS solve.
That is all the IRLS driver ever needs, so the statistics stay decoupled from
the linear algebra.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


#: <strategy_protocol>
class ReweightingStrategy(ABC):
    """Supplies the working response z and weights w for the next P-WLS solve."""

    is_glm: bool = False
    name: str = "strategy"

    def initial_eta(self, y: torch.Tensor) -> torch.Tensor:
        """Starting linear predictor. Default (identity link): the response itself."""
        return y.clone()

    @abstractmethod
    def working_response_and_weights(
        self, y: torch.Tensor, eta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z, w) given the current linear predictor eta."""

    def inverse_link(self, eta: torch.Tensor) -> torch.Tensor:
        """Map the linear predictor to the mean scale (identity by default)."""
        return eta

    def mean_objective(self, y: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """Scalar objective for convergence / step-halving. Must stay a mean so it
        scales with the penalized roughness term."""
        residual = y - eta
        return (residual * residual).mean()
#: </strategy_protocol>
