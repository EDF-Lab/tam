# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Implements the Linear Tree (Varying-Coefficient) Effect."""

import torch
import numpy as np
from typing import List, Optional

from ._base_effects import BaseEffect
from ._tensor import TensorProductEffect
from ._tree import TreeEffect
from ._linear import LinearEffect

class LinearTreeEffect(BaseEffect):
    r"""
    Implements a Linear Tree (Varying-Coefficient Tree).
    
    Mathematically represents:
    y = Tree_Base(X) + (Tree_Slope(X) * X_linear)
    """

    def __init__(
        self, 
        feature_name: str, 
        slope_feature: str,
        n_trees: int, 
        max_depth: int,
        max_leaves: Optional[int], 
        lambda_p: float,
        additional_features: Optional[List[str]],
        seed: int,
        extrapolate: str,
        sparsity_alpha: float = 0.0,
        split_strategy: str = "uniform"
    ):
        super().__init__(feature_name, "linear_tree", lambda_p, extrapolate)
        
        self.base_tree = TreeEffect(
            feature_name, n_trees, max_depth, max_leaves, lambda_p, additional_features, seed, 'continue', 
            sparsity_alpha, split_strategy
        )

        self.slope_tree = TreeEffect(
            feature_name, n_trees, max_depth, max_leaves, 1.0, additional_features, seed, 'continue', 
            sparsity_alpha, split_strategy
        )
        self.linear = LinearEffect(slope_feature, scaled=np.pi, lambda_p=1.0, extrapolate='continue')
        self.tensor = TensorProductEffect([self.slope_tree, self.linear], lambda_p, 'continue')
        
        self.tree_features = getattr(self.base_tree, 'input_features', [feature_name])
        self.input_features = self.tree_features + [slope_feature]

    def get_n_coeffs(self) -> int:
        return self.base_tree.get_n_coeffs() + self.tensor.get_n_coeffs()

#: <feature_map>
    def build_feature_map(self, x_data: torch.Tensor) -> torch.Tensor:

        n_tree_cols = len(self.tree_features)
        x_tree = x_data[..., 0 : n_tree_cols]
        
        phi_base = self.base_tree.transform(x_tree)
        phi_slope_tree = self.slope_tree.transform(x_tree)
        
        x_linear = x_data[..., n_tree_cols : n_tree_cols + 1].squeeze(-1) 
        phi_linear = self.linear.transform(x_linear)
        
        phi_tensor = TensorProductEffect.kronecker_product_einsum(phi_slope_tree, phi_linear)
        
        return torch.cat([phi_base, phi_tensor], dim=-1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        
        P1 = self.base_tree.build_penalty_matrix()
        P2 = self.tensor.build_penalty_matrix()
        
        n1, n2 = P1.shape[0], P2.shape[0]
        
        indices_list = []
        values_list = []
        
        # Block 1: Base Tree Intercepts (Top Left)
        if P1.is_sparse:
            P1 = P1.coalesce()
            indices_list.append(P1.indices())
            values_list.append(P1.values())
        else:
            nz = P1.nonzero(as_tuple=True)
            if nz[0].numel() > 0:
                indices_list.append(torch.stack([nz[0], nz[1]], dim=0))
                values_list.append(P1[nz])
                
        # Block 2: Tensor Product Slopes (Bottom Right)
        if P2.is_sparse:
            P2 = P2.coalesce()
            indices_list.append(P2.indices() + n1)
            values_list.append(P2.values())
        else:
            nz = P2.nonzero(as_tuple=True)
            if nz[0].numel() > 0:
                indices_list.append(torch.stack([nz[0] + n1, nz[1] + n1], dim=0))
                values_list.append(P2[nz])
                
        # Safely concatenate and build the global sparse block
        if indices_list:
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list, dim=0)
            return torch.sparse_coo_tensor(indices, values, size=(n1+n2, n1+n2), device=P1.device)
        else:
            return torch.sparse_coo_tensor(
                size=(n1+n2, n1+n2), dtype=P1.dtype, device=P1.device
            )
#: </penalty_matrix>