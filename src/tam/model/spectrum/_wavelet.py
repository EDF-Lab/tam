# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Implements Wavelet effects (Time-Frequency analysis)."""

import torch
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_wavelet>
class WaveletEffect(BaseEffect):
    r"""
    Implements a Wavelet effect using the Ricker wavelet.

    Unlike Fourier basis functions which are global, wavelets are localized 
    in both time and frequency. This effect generates a grid of wavelets 
    shifted in time (translation tau) and dilated in width (scale s).

    Attributes:
        n_scales (int): Number of distinct width levels.
        n_locations (int): Number of time centers per scale.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_scales: int, 
        n_locations: int, 
        lambda_p: float,
        extrapolate: str
    ):
        """Initializes the wavelet effect."""
        super().__init__(feature_name, "wavelet", lambda_p, extrapolate)
        self.n_scales = n_scales
        self.n_locations = n_locations
        
        self.scale_factors = torch.pow(
            2.0, 
            -torch.arange(n_scales, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        )
        
        self.locations = None
        self.real_scales = None

    def get_n_coeffs(self) -> int:
        return self.n_scales * self.n_locations

    def _init_grid(self, x_data: torch.Tensor, is_dummy: bool = False):
        """Initializes the time-scale grid."""
        x_min, x_max = x_data.min(), x_data.max()
        span = x_max - x_min
        if span == 0: span = 1.0
        
        locs = torch.linspace(
            x_min, x_max, self.n_locations, 
            device=x_data.device, dtype=torch.get_default_dtype()
        )
        r_scales = self.scale_factors.to(x_data.device) * (span * 0.2)
        
        # Safeguard: Do not permanently save the grid if probed by the memory estimator
        if not is_dummy:
            self.locations = locs
            self.real_scales = r_scales
            
        return locs, r_scales
#: </init_wavelet>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Builds the pure-tensor Wavelet feature map."""
        # Detect if this is a 1-sample dummy tensor used for memory estimation
        is_dummy = x_col.shape[-1] == 1 if x_col.dim() > 0 else False
        
        if self.locations is None:
            locs, r_scales = self._init_grid(x_col, is_dummy)
        else:
            locs, r_scales = self.locations, self.real_scales
            
        # --- Full Tensor Broadcasting Fix (No CPU/GPU cat memory spikes) ---
        # Expand x_col for broadcasting: (..., N) -> (..., N, 1, 1)
        x_expanded = x_col.unsqueeze(-1).unsqueeze(-1)
        
        # Scales: (S) -> (1, S, 1)
        s = r_scales.view(1, -1, 1)
        
        # Locations: (L) -> (1, 1, L)
        locs_expanded = locs.view(1, 1, -1)
        
        # Compute the entire Time-Scale grid simultaneously on GPU
        z = (x_expanded - locs_expanded) / s
        psi = (1 - z**2) * torch.exp(-0.5 * z**2) / torch.sqrt(s)
            
        return psi.reshape(*x_col.shape, -1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Builds the scale-dependent penalty matrix.
        """
        blocks = []
        
        for s_factor in self.scale_factors:
            weight = self.lambda_p / (s_factor ** 2)
            block = torch.ones(
                self.n_locations, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
            ) * weight
            blocks.append(block)
            
        diag = torch.cat(blocks)
        return torch.diag(diag)
#: </penalty_matrix>