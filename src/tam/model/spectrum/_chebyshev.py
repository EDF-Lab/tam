# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Implements Chebyshev polynomial effects.

This module provides the `ChebyshevEffect` class, which projects input data
onto a basis of Chebyshev polynomials of the first kind T_n.
This is optimal for modeling smooth, non-periodic global trends on finite 
intervals while minimizing Runge's phenomenon.
"""

import torch
from ._base_effects import BaseEffect
from tam.common.utils import TORCH_DEVICE

#: <init_chebyshev>
class ChebyshevEffect(BaseEffect):
    r"""
    Implements a global trend effect using Chebyshev Polynomials.

    Standard polynomial regression is numerically unstable and suffers from 
    Runge's phenomenon (divergence at edges). Chebyshev polynomials (T_n) 
    minimize the maximum interpolation error on [-1, 1].

    Attributes:
        degree (int): The maximum polynomial degree M.
        s (int): Sobolev-like regularization order.
    """

    def __init__(self, feature_name: str, degree: int, s: int, lambda_p: float, extrapolate: str):
        r"""Initializes the Chebyshev effect.

        Args:
            feature_name: Feature column name.
            degree: Maximum degree of the polynomial basis (excluding intercept).
            s: Spectral penalty order (0=Ridge, >0=Smoothness).
            lambda_p: Regularization strength.
        """
        super().__init__(feature_name, "chebyshev", lambda_p, extrapolate)
        self.degree = degree
        self.s = s

    def get_n_coeffs(self) -> int:
        return self.degree
#: </init_chebyshev>

#: <feature_map>
    def build_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Builds the Chebyshev design matrix via stable recurrence.
        
        Args:
            x: Input data. Must be normalized to [-1, 1].
        
        Returns:
            Design matrix of shape (..., n_samples, degree).
        """
        
        # Pre-allocate the full tensor to avoid `cat` memory spikes
        out_shape = list(x.shape) + [self.degree]
        phi_out = torch.empty(out_shape, dtype=torch.get_default_dtype(), device=x.device)
        
        # Fill iteratively in-place
        t_n_minus_1 = torch.ones_like(x) 
        t_n = x
        phi_out[..., 0] = t_n
        
        for i in range(1, self.degree):
            t_next = 2 * x * t_n - t_n_minus_1
            phi_out[..., i] = t_next
            t_n_minus_1 = t_n
            t_n = t_next
            
        return phi_out
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Builds a diagonal spectral penalty matrix.
        P_{kk} = lambda_p * (1 + k^{2s})
        """
        degrees = torch.arange(1, self.degree + 1, device=TORCH_DEVICE)
        diag_values = self.lambda_p * (1 + degrees**(2 * self.s))
        return torch.diag(diag_values)
#: </penalty_matrix>