r"""Implements the Fourier effect with Sobolev regularization.

Projects data onto a truncated Fourier basis.
"""

import torch
import numpy as np
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_fourier>
class FourierEffect(BaseEffect):
    r"""
    Implements a cyclical effect using a Fourier Series.
    
    Inputs are expected in [-1, 1] and are internally rescaled to [-pi, pi].
    If cyclic=True, it enforces a strict periodic boundary.
    """

    def __init__(self, feature_name: str, m: int, s: int, lambda_p: float, cyclic: bool, extrapolate: str):
        if m <= 0 or s < 0:
            raise ValueError(f"Invalid params for FourierEffect: m={m}, s={s}")
            
        super().__init__(feature_name, "fourier", lambda_p, extrapolate)
        
        self.m = int(m)
        self.s = int(s)
        self.cyclic = cyclic

    def get_n_coeffs(self) -> int:
        return 2 * self.m
#: </init_fourier>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""Builds Fourier basis functions."""
        x_scaled = x_col * np.pi
        x_expanded = x_scaled.unsqueeze(-1)
        
        freqs = torch.arange(1, self.m + 1, device=x_col.device, dtype=torch.get_default_dtype())
        dims_to_add = x_expanded.dim() - 1
        freqs_expanded = freqs.view(*([1] * dims_to_add), -1)
        
        if self.cyclic:
            theta = x_expanded * freqs_expanded
        else:
            theta = x_expanded * freqs_expanded / 2
            
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Diagonal Sobolev penalty for Real Fourier Basis."""
        freqs = torch.arange(1, self.m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        penalty_half = self.lambda_p * (1 + freqs ** (2 * self.s))
        diag_full = torch.cat([penalty_half, penalty_half])
        return torch.diag(diag_full)
#: </penalty_matrix>