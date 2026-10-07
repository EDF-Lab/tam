# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements Tensor Product effects (Interactions)."""

import torch
import functools
from typing import List, Optional
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_tensor>
class TensorProductEffect(BaseEffect):
    r"""
    Implements an interaction (Tensor Product) between multiple effects.

    This effect computes the Kronecker product of the feature maps of its
    component effects.

    The penalty is anisotropic, meaning smoothness can be controlled
    independently in each direction:

    Attributes:
        effects (List[BaseEffect]): The list of sub-effects to cross.
    """

    def __init__(self, effects: List[BaseEffect], lambda_p: float, extrapolate: str, fold_weight: bool = True):
        r"""
        Initializes the tensor product effect.
        
        Args:
            effects: List of initialized BaseEffect objects (e.g., [Spline(x), Spline(y)]).
            lambda_p: Global regularization scaling factor for the interaction surface.
            fold_weight: Fold ``lambda_p`` into the margins (one smoothing parameter per margin, the default). ``False`` keeps
                it as the single weight of the tensor product, as a ``LinearTreeEffect`` needs for its slope surface.

        """
        # Name example: "te_s_temp_x_f_hour"
        name = "te_" + "_x_".join([e.feature_name for e in effects])
        # One smoothing parameter per margin (mgcv semantics): the weight of the tensor product itself, formula default or te-level
        # `ap`, is folded into each margin once, so the penalty matrix of fixed weights is the one it always was.
        if fold_weight:
            for margin in effects:
                if margin.n_penalty_coordinates == 1:
                    margin.lambda_p = float(lambda_p) * margin.lambda_p
            lambda_p = 1.0
        super().__init__(name, "tensor_product", lambda_p, extrapolate)
        self.effects = effects
        self._fold_weight = fold_weight

    def penalty_coordinates(self) -> list:
        r"""The smoothing weight of each margin: ``sum_i lambda_i (I x ... x P_i x ... x I)``."""
        if not self._fold_weight:
            return [self.lambda_p]
        coordinates = []
        for margin in self.effects:
            if margin.n_penalty_coordinates != 1:
                raise NotImplementedError("A tensor product nested in a tensor product has no per-margin smoothing coordinates.")
            coordinates.append(margin.lambda_p)
        return coordinates

    def set_penalty_coordinates(self, values) -> None:
        values = list(values)
        if not self._fold_weight:
            return super().set_penalty_coordinates(values)
        if len(values) != len(self.effects):
            raise ValueError(f"The tensor product {self.feature_name} has {len(self.effects)} penalty coordinates, got {len(values)}.")
        for margin, value in zip(self.effects, values):
            margin.lambda_p = float(value)

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Initialises each margin from its columns, sliced as in ``build_feature_map``."""
        col_idx = 0
        for effect in self.effects:
            n_cols = len(getattr(effect, 'input_features', [effect.feature_name]))
            effect.initialize(x_cols[..., col_idx : col_idx + n_cols])
            col_idx += n_cols

    def get_n_coeffs(self) -> int:
        r"""
        Computes the total number of coefficients (product of dimensions).
        Example: 10 knots * 12 hours = 120 coefficients.
        """
        dims = [e.get_n_coeffs() for e in self.effects]
        return functools.reduce(lambda x, y: x * y, dims)
#: </init_tensor>

#: <feature_map>
    @staticmethod
    def kronecker_product_einsum(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kronecker Product pair by pair using Einstein summation broadcasting.
        Exposed statically so composite effects (like LinearTree) can use it natively.
        """
        t1_expanded = t1.unsqueeze(-1) # (..., d1, 1)
        t2_expanded = t2.unsqueeze(-2) # (..., 1, d2)
        out = t1_expanded * t2_expanded
        target_dim = t1.shape[-1] * t2.shape[-1]
        return out.reshape(*out.shape[:-2], target_dim)

    def build_feature_map(self, x_data: torch.Tensor) -> torch.Tensor:
        r"""
        Builds the global feature map via Kronecker product.
        
        Args:
            x_data: Input tensor (Batch, N_samples, N_effects).
        
        Note: The factory ensures columns are correctly ordered/selected.
        """
        #  Generate feature maps for each sub-effect
        phi_list = []
        col_idx = 0
        for effect in self.effects:
            # Determine how many columns this sub-effect needs
            n_cols = len(getattr(effect, 'input_features', [effect.feature_name]))
            
            # Slice the exact number of columns required
            x_cols = x_data[..., col_idx : col_idx + n_cols]
            
            # Only squeeze the tensor for legacy 1D math effects. 
            # Spatial kernels strictly require the feature dimension to remain intact.
            if n_cols == 1 and effect.__class__.__name__ not in ['TreeEffect', 'LinearTreeEffect', 'RBFEffect', 'NeuralEffect']:
                x_cols = x_cols.squeeze(-1)
                
            phi_list.append(effect.transform(x_cols))
            col_idx += n_cols

        # Reduce the list of tensors into a single tensor
        phi_cross = functools.reduce(self.kronecker_product_einsum, phi_list)
        
        return phi_cross
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Builds the anisotropic penalty matrix.
        
        It sums the penalties of each dimension, expanded by Identity matrices
        on other dimensions. Each margin penalty carries the weight lambda_i of its own
        coordinate, so GCV tunes one smoothing parameter per margin.
        """
        dims = [e.get_n_coeffs() for e in self.effects]
        penalties = [e.build_penalty_matrix() for e in self.effects]
        
        P_total = 0
        
        for i in range(len(self.effects)):
            # Construct term i: I x ... x P_i x ... x I
            current_term = None
            
            for j in range(len(self.effects)):
                if i == j:
                    mat = penalties[j] # Active penalty
                else:
                    mat = torch.eye(
                        dims[j], device=TORCH_DEVICE, dtype=torch.get_default_dtype()
                    ) # Identity (Passive)
                
                if mat.is_sparse:
                    mat = mat.to_dense()
                    
                if current_term is None:
                    current_term = mat
                else:
                    # Kronecker product of matrices
                    current_term = torch.kron(current_term, mat)
            
            P_total = P_total + current_term
            
        # Each P_i already carries its margin's lambda_i; the weight of the tensor product itself stays 1
        return P_total * self.lambda_p
#: </penalty_matrix>