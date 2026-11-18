# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Hurst fractional-Gaussian-noise spectrum effect.

A fixed fractional-smoothness prior on pointwise harmonics (Mandelbrot & Van Ness 1968): the spectral precision

.. math::

    P_{\text{hurst}} = \operatorname{diag}\big( |\omega_j|^{2H - 1} \big)_{j=1}^k \cdot \lambda_p

with the Hurst exponent :math:`H` given, not estimated.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import from_btk, to_btd


class HurstEffect(BaseEffect):
    r"""Fractional Brownian Motion (fBm) / fractional Gaussian noise (fGn) effect.

    Penalizes Fourier modes by the inverse power spectral density of fractional
    Gaussian noise :math:`S_H(\omega) \propto |\omega|^{1 - 2H}` (Mandelbrot & Van Ness 1968):

    .. math::

        P_{\text{hurst}} = \operatorname{diag}\big( |\omega_j|^{2H - 1} \big)_{j=1}^k \cdot \lambda_p

    where :math:`H \in (0, 1)` is the Hurst exponent:
    - :math:`H > 0.5`: persistent long-memory dynamics with power-law autocorrelation decay.
    - :math:`H = 0.5`: standard memoryless Brownian motion increments (flat Ridge penalty).
    - :math:`H < 0.5`: anti-persistent, rough mean-reverting dynamics.

    Args:
        feature_name (str): Input time series feature name.
        k (int): Total number of Fourier coefficients (must be even). Defaults to 16.
        H (float): Hurst parameter :math:`H \in (0, 1)`. Defaults to 0.75.
        period (Optional[float]): Base period length. Defaults to data span.
        lambda_p (float): Regularization penalty strength. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = False

    def __init__(
        self,
        feature_name: str,
        k: int = 16,
        H: float = 0.75,
        period: Optional[float] = None,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "hurst", lambda_p, extrapolate)
        self.k = max(2, int(k))
        if self.k % 2 != 0:
            self.k += 1
        self.m = self.k // 2
        self.H = float(np.clip(H, 0.05, 0.95))
        self.period = float(period) if period is not None else None
        self._fitted_period: Optional[float] = None

    def initialize(self, x_col: torch.Tensor) -> None:
        """Caches the empirical training period span if period is not explicitly fixed."""
        if self.period is not None:
            self._fitted_period = self.period
            return
        x = x_col.reshape(-1)
        if x.numel() > 0:
            span = float(x.max().item() - x.min().item())
            self._fitted_period = max(span, 1e-3)

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"H={self.H:.2f}, k={self.k}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.k

#: <hurst_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Constructs harmonic Fourier basis functions of each value: ``(B, T)`` in, ``(B, T, k)`` out (pointwise, no state)."""
        x3, shape, layout = to_btd(x_col, False)
        x = x3[..., 0].reshape(-1)
        device = x_col.device
        dtype = x_col.dtype

        # Period from the frozen training span if not explicitly fixed
        if self.period is not None:
            P = self.period
        elif self._fitted_period is not None:
            P = self._fitted_period
        else:
            self.initialize(x_col)
            P = self._fitted_period if self._fitted_period is not None else 1.0

        freqs = torch.arange(1, self.m + 1, device=device, dtype=dtype)  # (m,)
        phases = (2.0 * np.pi / P) * torch.outer(x, freqs)  # (N, m)

        phi = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (N, 2m)
        return from_btk(phi.reshape(x3.shape[0], x3.shape[1], self.k), shape, layout)
#: </hurst_feature_map>

#: <hurst_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs inverse fGn spectral precision penalty:

        .. math::

            P = \lambda_p\, I_2 \otimes \operatorname{diag}\big( j^{2H - 1} \big)_{j=1}^m
        """
        j_vec = torch.arange(1, self.m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        # Power law weight j^(2H - 1)
        power_weights = j_vec ** (2.0 * self.H - 1.0)
        # Duplicate for sin and cos
        diag_weights = torch.cat([power_weights, power_weights])
        return torch.diag(diag_weights) * self.lambda_p
#: </hurst_penalty_matrix>
