# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements categorical effects via One-Hot Encoding."""

import torch
import numpy as np
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_categorical>
class CategoricalEffect(BaseEffect):
    r"""
    Categorical effect supporting Nominal (Ridge) and Ordinal (Difference) topologies.

    A code that is none of the levels (outside ``0 .. n_categories - 1`` once normalised) gives the zero row: the effect is 0 there and
    the prediction falls back on the intercept and the other terms. A term with a single level (a flag that is constant in the training
    rows) keeps that level for every row: its column is then the constant of the intercept, and dropping it would remove part of the intercept. ``vocabulary`` is set when the levels are the codes of the training
    rows (``n_cat`` omitted in the formula); the model then maps the codes to level indices before the effect sees them.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_categories: int, 
        topology: str, 
        lambda_p: float,
        penalty_order: int,
        extrapolate: str
    ):
        clean_topo = topology.lower()
        if clean_topo not in ['nominal', 'ordinal', 'fourier']:
            raise ValueError(f"Topology '{topology}' invalid. Use 'nominal' or 'ordinal'.")

        super().__init__(feature_name, f"categorical_{clean_topo}", lambda_p, extrapolate)
        self.n_categories = int(n_categories)
        self.vocabulary = False
        self.topology = clean_topo
        self.penalty_order = penalty_order

        if self.topology == 'fourier':
            self.m = self.n_categories // 2 + self.n_categories % 2
            self.s = 0

    def get_n_coeffs(self) -> int:
        if self.topology == 'fourier':
            return 2 * self.m
        return self.n_categories
    
#: </init_categorical>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Projects inputs based on topology:
        - Nominal/Ordinal: One-Hot basis.
        - Fourier: Sin/cos basis.
        """

        if self.topology in ['nominal', 'ordinal']:
            x_indices = torch.round((x_col + 1.0) / 2.0 * (self.n_categories - 1)).long()
            if self.n_categories == 1:
                known = torch.ones_like(x_col, dtype=torch.bool)  # a single level (a feature constant in training) is every row, as before
            else:
                known = (x_indices >= 0) & (x_indices <= self.n_categories - 1)
            x_safe = torch.clamp(x_indices, 0, self.n_categories - 1)
            phi = torch.nn.functional.one_hot(x_safe, num_classes=self.n_categories)
            phi = phi * known.unsqueeze(-1).to(phi.dtype)  # a code outside the levels is the zero row: no categorical effect
            return phi.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        
        elif self.topology == 'fourier':
            x_scaled = x_col * np.pi
            x_expanded = x_scaled.unsqueeze(-1)
            freqs = torch.arange(1, self.m + 1, device=x_col.device, dtype=torch.get_default_dtype())
            dims_to_add = x_expanded.dim() - 1
            freqs_expanded = freqs.view(*([1] * dims_to_add), -1)
            theta = x_expanded * freqs_expanded / 2
            return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)    
        
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds penalty: Identity (Nominal) or Difference (Ordinal)."""
        if self.topology == 'nominal':
            return torch.eye(
                self.n_categories, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
            ) * self.lambda_p
            
        elif self.topology == 'ordinal':
            # Create the identity matrix directly on the GPU
            I = torch.eye(self.n_categories, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            # Compute differences natively on the GPU
            D = torch.diff(I, n=self.penalty_order, dim=0)
            P = D.T @ D
            return self.lambda_p * P
        
        elif self.topology == 'fourier':            
            freqs = torch.arange(1, self.m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            penalty_half = self.lambda_p * (1 + freqs ** (2 * self.s))
            diag_full = torch.cat([penalty_half, penalty_half])
            return torch.diag(diag_full)
        
        return torch.zeros(1)
#: </penalty_matrix>