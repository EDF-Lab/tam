# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France) et Sorbonne Université
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-License-Identifier: LGPL-3.0-or-later
# Authors : Yann Allioux, Nathan Doumèche

r"""Implements linear and offset effects.

Contains the simplest additive blocks: global intercept and linear regressors.
"""

from typing import Tuple
import torch
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <offset_effect>
class OffsetEffect(BaseEffect):
    r"""Implements the global Offset (constant bias)."""
    
    def __init__(self, lambda_p: float, extrapolate: str):
        super().__init__("offset", "offset", lambda_p, extrapolate)

    def get_n_coeffs(self) -> int:
        return 1

    def build_feature_map(self, x_data: torch.Tensor) -> torch.Tensor:
        r"""Generates a tensor of ones matching the batch shape."""
        x_shape_prefix = x_data.shape[:-1]
        return torch.ones(*x_shape_prefix, 1, device=x_data.device, dtype=torch.get_default_dtype())

    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Ridge penalty for the offset (usually small)."""
        return torch.tensor([[self.lambda_p]], device=TORCH_DEVICE, dtype=torch.get_default_dtype())
#: </offset_effect>


#: <linear_effect>
class LinearEffect(BaseEffect):
    r"""Implements a simple Linear effect mapping directly to the scaled input space."""

    def __init__(self, feature_name: str, scaled: float, lambda_p: float, extrapolate: str):
        super().__init__(feature_name, "linear", lambda_p, extrapolate)
        self.scaled = scaled

    def get_n_coeffs(self) -> int:
        return 1

    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""Reshapes the input data to be a column vector."""
        x_scaled = x_col * self.scaled
        return x_scaled.unsqueeze(-1)

    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Ridge penalty."""
        return torch.tensor([[self.lambda_p]], device=TORCH_DEVICE, dtype=torch.get_default_dtype())
#: </linear_effect>