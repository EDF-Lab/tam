# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Implements Universal Physics Effects (PIKL/PC-GAM).

This module allows constraining a statistical model with an arbitrary linear
differential operator (PDE).
"""

import torch
import numpy as np
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

from ._spline import SplineEffect
from ._fourier import FourierEffect
from ._neural import NeuralEffect

#: <init_physics>
class UniversalPhysicsEffect(BaseEffect):
    """
    Universal Physics Effect constrained by a linear PDE.

    This effect imposes that the learned function f minimizes the residual
    of a differential equation L(f) approx 0.

    The operator L is defined by a weighted sum of derivatives:
    L(f) = w_0 * f + w_1 * df/dt + w_2 * d^2f/dt^2 + ...

    Attributes:
        basis_type (str): The type of basis used for approximation.
        diff_weights (dict): Dictionary of operator weights.
    """

    def __init__(
        self,
        feature_name: str,
        basis_type: str,
        n_coeffs: int,
        diff_weights: dict,
        lambda_p: float,
        extrapolate: str,
        **basis_params
    ):
        super().__init__(feature_name, f"phys_{basis_type}", lambda_p, extrapolate)
        self.basis_type = basis_type
        self.diff_weights = diff_weights
        
        if basis_type == 'spline':
            deg = basis_params.get('spline_degree', 3)
            self.base_effect = SplineEffect(
                feature_name, n_knots=n_coeffs, spline_degree=deg, penalty_order=0, lambda_p=0.0, extrapolate='continue'
            )
            self.n_coeffs_val = self.base_effect.get_n_coeffs()
            
        elif basis_type == 'fourier':
            m = n_coeffs // 2
            s_val = basis_params.get('s', 0)
            cyclic = basis_params.get('cyclic', False)
            self.base_effect = FourierEffect(
                feature_name, m=m, s=s_val,cyclic=cyclic,
                lambda_p=0.0, extrapolate='continue')
            self.n_coeffs_val = self.base_effect.get_n_coeffs()
            
        elif basis_type == 'neural':
            act = basis_params.get('act', 'tanh')
            seed = basis_params.get('seed', 42)
            layers = basis_params.get('n_hidden_layers', 1)
            others_str = basis_params.get('others', None)
            additional_features = [s.strip() for s in str(others_str).split('|') if s.strip()] if others_str else None
            self.base_effect = NeuralEffect(
                feature_name,
                n_neurons=n_coeffs,
                activation=act,
                additional_features=additional_features,
                seed=seed,
                n_hidden_layers=layers,
                lambda_p=0.0, extrapolate='continue'
            )
            self.n_coeffs_val = n_coeffs
        else:
            raise ValueError(f"Unknown basis type for physics effect: {basis_type}")

    keeps_feature_dim = property(lambda self: self.base_effect.keeps_feature_dim)

    @property
    def required_features(self) -> list:
        return self.base_effect.required_features

    def summary_type(self) -> str:
        return "Physics"

    def summary_details(self) -> str:
        ops = [f"{k}={v}" for k, v in self.diff_weights.items()]
        return f"Basis={self.basis_type}, PDE[{', '.join(ops)}]"

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Initialises the basis (knots, grid, network weights) on the training tensor; the penalty of a neural basis reads its weights."""
        self.base_effect.initialize(x_cols)
        self.n_coeffs_val = self.base_effect.get_n_coeffs()

    def get_n_coeffs(self) -> int:
        return self.n_coeffs_val

    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        return self.base_effect.build_feature_map(x_col)
#: </init_physics>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds the Stiffness Matrix P such that f^T P f approximates the integral of (L(f))^2."""
        if self.basis_type == 'spline':
            return self._build_spline_penalty()
        elif self.basis_type == 'fourier':
            return self._build_fourier_penalty()
        elif self.basis_type == 'neural':
            return self._build_neural_penalty()
        return torch.zeros(1)
#: </penalty_matrix>

#: <spline_penalty>
    def _build_spline_penalty(self) -> torch.Tensor:
        I = torch.eye(self.n_coeffs_val, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        has_fractional = any(not float(k[1:]).is_integer() for k in self.diff_weights.keys())
        
        # 1. Find the maximum derivative order to determine the valid interior size
        max_order = max([int(np.ceil(float(k[1:]))) for k in self.diff_weights.keys()])
        valid_rows = max(1, self.n_coeffs_val - max_order)
        n_knots = getattr(self.base_effect, "n_knots", max(1, self.n_coeffs_val - 3))
        dx = 2.0 / float(max(1, n_knots))
        
        L_op = torch.zeros((valid_rows, self.n_coeffs_val), device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        
        for key, weight in self.diff_weights.items():
            order = float(key[1:])
            scale_dx = float(dx ** (-order)) if has_fractional else 1.0
            if order == 0.0:
                mat = I
                diff_rows = mat.shape[0]
                if diff_rows > valid_rows:
                    crop_total = diff_rows - valid_rows
                    crop_top = crop_total // 2
                    mat = mat[crop_top : crop_top + valid_rows, :]
            elif order.is_integer():
                mat = torch.diff(I, n=int(order), dim=0)
                diff_rows = mat.shape[0]
                if diff_rows > valid_rows:
                    crop_total = diff_rows - valid_rows
                    crop_top = crop_total // 2
                    mat = mat[crop_top : crop_top + valid_rows, :]
                mat = mat * scale_dx
            else:
                # Grünwald-Letnikov fractional finite difference weights
                w_gl = torch.zeros(valid_rows, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
                w_gl[0] = 1.0
                for k in range(1, valid_rows):
                    w_gl[k] = w_gl[k - 1] * (1.0 - (order + 1.0) / float(k))
                mat_frac = torch.zeros((valid_rows, self.n_coeffs_val), device=TORCH_DEVICE, dtype=torch.get_default_dtype())
                for i in range(valid_rows):
                    length = min(i + 1, self.n_coeffs_val)
                    mat_frac[i, :length] = w_gl[:length].flip(dims=[0])
                # Caputo boundary condition: C_D^\alpha (1) = 0
                mat_frac = mat_frac - mat_frac.sum(dim=-1, keepdim=True) / float(self.n_coeffs_val)
                mat = mat_frac * scale_dx
                
            L_op += weight * mat
            
        return self.lambda_p * (L_op.T @ L_op)
#: </spline_penalty>

#: <fourier_penalty>
    def _build_fourier_penalty(self) -> torch.Tensor:
        m = self.base_effect.m
        freqs = torch.arange(1, m + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        eig_vals = torch.zeros_like(freqs, dtype=torch.complex128)

        for key, weight in self.diff_weights.items():
            order = float(key[1:])
            # an integer order uses the integer power, as in v1.5.1 (the complex power with a float exponent can differ by one unit in the last place between platforms)
            power = int(order) if order.is_integer() else order
            eig_vals += weight * (1j * freqs.to(torch.complex128)) ** power
            
        penalty_half = self.lambda_p * (torch.abs(eig_vals) ** 2).to(torch.get_default_dtype())
        diag_full = torch.cat([penalty_half, penalty_half])
        return torch.diag(diag_full)
#: </fourier_penalty>

#: <neural_penalty>
    def _build_neural_penalty(self) -> torch.Tensor:
        W = self.base_effect.weights.flatten()
        diag_vals = torch.zeros(self.n_coeffs_val, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        
        for key, weight in self.diff_weights.items():
            order = float(key[1:])
            term = weight * ((W ** int(order)) if order.is_integer() else (torch.abs(W) ** order))
            diag_vals += term
            
        return self.lambda_p * torch.diag(torch.abs(diag_vals)**2)
#: </neural_penalty>