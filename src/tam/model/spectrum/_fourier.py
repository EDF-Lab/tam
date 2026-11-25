# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the Fourier effect with Sobolev regularization.

Projects data onto a truncated Fourier basis.
"""

from typing import Optional, Tuple, Union

import torch
import numpy as np
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import eval_1d_dedup

#: <init_fourier>
class FourierEffect(BaseEffect):
    r"""
    Implements a cyclical effect using a Fourier Series.
    
    Inputs are expected in [-1, 1] and are internally rescaled to [-pi, pi].
    If cyclic=True, it enforces a strict periodic boundary.
    """

    def __init__(self, feature_name: str, m: int, s: int, lambda_p: float, cyclic: bool, extrapolate: str,
                 period: Optional[Union[str, Tuple[float, float]]] = None):
        """
        Args:
            period: The range of the input that makes one period. ``None`` (default): the training range (minimum, maximum),
                as before. ``"auto"``: (first value, last value + one step), the step being the smallest gap between the
                distinct training values, so hours 0..23 give (0, 24) and months 1..12 give (1, 13). ``(low, high)``: given,
                for a history that covers only part of the cycle (``(0, 1)`` for a time of year).
        """
        if m <= 0 or s < 0:
            raise ValueError(f"Invalid params for FourierEffect: m={m}, s={s}")
        if isinstance(period, str):
            if period != "auto":
                raise ValueError(f"FourierEffect period must be None, 'auto' or (low, high); got {period!r}")
        elif period is not None and not float(period[0]) < float(period[1]):
            raise ValueError(f"FourierEffect period must be (low, high) with low < high; got {period}")

        super().__init__(feature_name, "fourier", lambda_p, extrapolate)

        self.m = int(m)
        self.s = int(s)
        self.cyclic = cyclic
        self.period = period if period is None or isinstance(period, str) else (float(period[0]), float(period[1]))

    def summary_details(self) -> str:
        return f"m={self.m}, Sobolev(s={self.s})"

    def get_n_coeffs(self) -> int:
        return 2 * self.m
#: </init_fourier>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""Builds Fourier basis functions."""
        dtype = torch.get_default_dtype()
        freqs = torch.arange(1, self.m + 1, device=x_col.device, dtype=dtype)
        scale = np.pi if self.cyclic else (0.5 * np.pi)

        def _eval(xv: torch.Tensor) -> torch.Tensor:
            freqs_exp = freqs.view(*([1] * xv.dim()), -1)
            theta = (xv.unsqueeze(-1) * scale) * freqs_exp
            return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

        return eval_1d_dedup(x_col, _eval)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Diagonal Sobolev penalty for Real Fourier Basis."""
        freqs = torch.arange(1, self.m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        penalty_half = self.lambda_p * (1 + freqs ** (2 * self.s))
        diag_full = torch.cat([penalty_half, penalty_half])
        return torch.diag(diag_full)
#: </penalty_matrix>