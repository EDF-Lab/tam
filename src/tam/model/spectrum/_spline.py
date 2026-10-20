# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the P-Spline effect (B-Splines + Difference Penalty)."""

import torch
import numpy as np

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_knots>
class SplineEffect(BaseEffect):
    r"""
    Implements a P-Spline effect.
    
    Robust implementation with manual knot vector construction to handle 
    sparse or constant data distributions.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_knots: int, 
        spline_degree: int, 
        penalty_order: int, 
        lambda_p: float,
        extrapolate: str,
        m_pen: int = 1
    ):
        super().__init__(feature_name, "spline", lambda_p, extrapolate)
        self.n_knots = n_knots
        self.spline_degree = spline_degree
        self.penalty_order = penalty_order

        self._n_coeffs = self.n_knots + self.spline_degree
        self._cached_knots = None

        # m_pen > 1 splits the roughness penalty in m_pen locally weighted parts, each with its own smoothing parameter.
        self.m_pen = int(m_pen)
        n_differences = self._n_coeffs - self.penalty_order
        if self.m_pen < 1 or self.m_pen > max(n_differences, 1):
            raise ValueError(f"m_pen must be between 1 and the {max(n_differences, 1)} differences of the penalty of s({feature_name}); got {m_pen}.")
        self._part_lambdas = None

    def penalty_coordinates(self) -> list:
        r"""The smoothing weight of each part of the penalty (one per ``m_pen``)."""
        if self.m_pen == 1:
            return [self.lambda_p]
        return list(self._part_lambdas) if self._part_lambdas is not None else [self.lambda_p] * self.m_pen

    def set_penalty_coordinates(self, values) -> None:
        values = [float(v) for v in values]
        if len(values) != self.m_pen:
            raise ValueError(f"s({self.feature_name}) has {self.m_pen} penalty coordinates, got {len(values)}.")
        if self.m_pen == 1:
            self.lambda_p = values[0]
            return
        self._part_lambdas = values
        self.lambda_p = float(np.exp(np.mean(np.log([max(v, 1e-300) for v in values]))))  # the geometric mean stands for the effect where one weight is read

    def _part_weights(self, n_rows: int) -> torch.Tensor:
        r"""Weights ``W`` of the ``m_pen`` parts over the ``n_rows`` differences of the penalty: B-splines of degree ``min(3, m_pen - 1)``
        on the index of the difference, non-negative and adding up to one in every row (a partition of unity)."""
        dtype = torch.get_default_dtype()
        if self.m_pen == 1:
            return torch.ones(n_rows, 1, device=TORCH_DEVICE, dtype=dtype)
        degree = min(3, self.m_pen - 1)
        n_internal = self.m_pen - degree - 1
        internal = torch.linspace(0.0, 1.0, n_internal + 2, dtype=dtype)[1:-1]
        knots = torch.cat([torch.zeros(degree + 1, dtype=dtype), internal, torch.ones(degree + 1, dtype=dtype)])
        position = torch.linspace(0.0, 1.0, n_rows, dtype=dtype) if n_rows > 1 else torch.full((1,), 0.5, dtype=dtype)
        weights = self._cox_de_boor(position, knots, degree)
        return (weights / weights.sum(dim=-1, keepdim=True)).to(TORCH_DEVICE)

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Sets the knots from the full training tensor, so that the memory probe (one row per group) never decides them."""
        if self._cached_knots is None:
            self._get_knots(x_cols)

    def summary_details(self) -> str:
        return f"Knots={self.n_knots}, Diff(d={self.penalty_order})" + (f", m_pen={self.m_pen}" if self.m_pen > 1 else "")

    def get_n_coeffs(self) -> int:
        return self._n_coeffs

    def _get_knots(self, x_data: torch.Tensor, is_dummy: bool = False) -> torch.Tensor:
        r"""Manually constructs the full knot vector with correct padding.

        Once set from the training data, the knots are never recomputed: a prediction must not depend on how many values
        it is evaluated on. A single value (one row per group, or the lone out-of-range value of a frame) is evaluated on
        the trained knots like any other.
        """
        if self._cached_knots is not None:
            return self._cached_knots

        valid_x = x_data[torch.isfinite(x_data)]
        if len(valid_x) == 0:
            x_min, x_max = 0.0, 1.0
        else:
            x_min, x_max = valid_x.min().item(), valid_x.max().item()
        
        if np.isclose(x_min, x_max):
            x_max += 1e-3
            x_min -= 1e-3

        internal_knots = torch.linspace(
            x_min, x_max, self.n_knots + 1, 
            device=x_data.device, dtype=torch.get_default_dtype()
        )[1:-1]
        
        k = self.spline_degree
        t_left = torch.full((k + 1,), x_min, device=x_data.device, dtype=torch.get_default_dtype())
        t_right = torch.full((k + 1,), x_max, device=x_data.device, dtype=torch.get_default_dtype())
        
        knots = torch.cat([t_left, internal_knots, t_right])
        
        if not is_dummy:
            self._cached_knots = knots
            self._n_coeffs = len(self._cached_knots) - (self.spline_degree + 1)

        return knots
#: </init_knots>

#: <cox_de_boor>
    def _cox_de_boor(self, x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
        r"""
        Evaluates B-spline basis functions using fully vectorized Cox-de Boor.
        """
        knots = knots.to(x.dtype)
        
        x_exp = x.unsqueeze(-1)
        
        x_c = torch.clamp(x_exp, min=knots[degree], max=knots[-(degree+1)] - 1e-6)
        
        bases = ((x_c >= knots[:-1]) & (x_c < knots[1:])).to(x.dtype)
        
        for d in range(1, degree + 1):
            dt_left = knots[d:-1] - knots[:-d-1]
            left_mask = dt_left > 1e-8
            left_term = torch.zeros_like(bases[..., :-1])
            
            left_term[..., left_mask] = (
                (x_c - knots[:-d-1])[..., left_mask] / dt_left[left_mask]
            ) * bases[..., :-1][..., left_mask]
            
            dt_right = knots[d+1:] - knots[1:-d]
            right_mask = dt_right > 1e-8
            right_term = torch.zeros_like(bases[..., 1:])
            
            right_term[..., right_mask] = (
                (knots[d+1:] - x_c)[..., right_mask] / dt_right[right_mask]
            ) * bases[..., 1:][..., right_mask]
            
            bases = left_term + right_term
            
        return bases
#: </cox_de_boor>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""Builds the pure-PyTorch B-spline design matrix."""
        x_shape = x_col.shape

        is_dummy = (x_col.shape[-1] == 1 if x_col.dim() > 0 else False)
        
        knots = self._get_knots(x_col, is_dummy=is_dummy)
        phi_matrix = self._cox_de_boor(x_col, knots, self.spline_degree)
        
        return phi_matrix.reshape(*x_shape, -1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds the finite difference penalty matrix entirely on GPU."""
        n_coeffs = self.get_n_coeffs()
        
        I = torch.eye(n_coeffs, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        
        if self.penalty_order == 0:
            D = I
        else:
            D = torch.diff(I, n=self.penalty_order, dim=0)

        if self.m_pen == 1:
            M_star_M = self.lambda_p * (D.T @ D)
            return M_star_M
        weights = self._part_weights(D.shape[0])
        lambdas = self.penalty_coordinates()
        return sum(lambdas[r] * (D.T @ (weights[:, r:r + 1] * D)) for r in range(self.m_pen))
#: </penalty_matrix>