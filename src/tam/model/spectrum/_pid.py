#: <module_doc>
r"""
Proportional-Integral-Derivative (PID) Effect.

This module provides the `PIDEffect` class, which transforms an autoregressive 
target lag into a discrete PID controller. It explicitly models short-term 
momentum and long-term accumulation, ensuring physically stable control dynamics.
"""
#: </module_doc>

import torch
import torch.nn.functional as F
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_pid>
class PIDEffect(BaseEffect):
    r"""
    Implements a discrete Proportional-Integral-Derivative (PID) constraint.

    By constraining standard autoregressive lags into a PID representation, 
    the Primal Solver enforces structural stability on the time-series memory.
    The integral term uses a rolling mean to maintain scale consistency with 
    the L2 (Ridge) penalty.

    Attributes:
        window (int): The look-back period for the Integral rolling mean.
        d_penalty_multiplier (float): Artificially boosts the Ridge penalty on the 
            Derivative term to prevent amplification of high-frequency stochastic noise.
    """

    def __init__(self, feature_name: str, window: int, lambda_p: float, d_penalty_multiplier: float, extrapolate: str):
        super().__init__(feature_name, "pid", lambda_p, extrapolate)
        
        if window < 0:
            raise ValueError(f"PID window must be non-negative. Got: {window}")
            
        self.window = int(window)
        self.d_penalty_multiplier = float(d_penalty_multiplier)

    def get_n_coeffs(self) -> int:
        """Returns the number of parameters learned by this effect (K_p, K_i, K_d)."""
        return 3
#: </init_pid>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Constructs the discrete PID feature space using vectorized tensor operations.

        Args:
            x_col (torch.Tensor): The raw autoregressive lag tensor. 
                                  Expected shape: (..., Time).

        Returns:
            torch.Tensor: The concatenated feature map [P, I, D]. 
                          Shape: (..., Time, 3).
        """
        # 1. Proportional (P): The raw lag state
        P = x_col.unsqueeze(-1)

        # 2. Derivative (D): Rate of change (y_t - y_{t-1})
        # Uses native torch.diff, prepending a zero to maintain the time dimension size
        zero_prep = torch.zeros_like(x_col[..., :1])
        D = torch.diff(x_col, dim=-1, prepend=zero_prep).unsqueeze(-1)

        # 3. Integral (I): Rolling mean over the specified 'window'
        cumsum_x = torch.cumsum(x_col, dim=-1)
        w = min(self.window, x_col.shape[-1])
        
        if w > 0:
            # Pad the left side with zeros to shift the cumulative sum by 'w' steps
            shift_cumsum = F.pad(cumsum_x[..., :-w], pad=(w, 0), mode='constant', value=0.0)
            I_val = cumsum_x - shift_cumsum
        else:
            I_val = cumsum_x

        # Divide by 'w' to convert the rolling sum into a rolling mean, 
        # guaranteeing the L2 penalty applies isotropically across P, I, and D.
        I = (I_val / max(1, w)).unsqueeze(-1)

        # 4. Assemble the final feature block
        return torch.cat([P, I, D], dim=-1)
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Constructs the stiffness matrix for the PID terms.

        Returns:
            torch.Tensor: A 3x3 diagonal penalty matrix. The Derivative term 
                          receives a boosted penalty to enforce low-pass filtering.
        """
        diag = torch.tensor(
            [1.0, 1.0, self.d_penalty_multiplier], 
            dtype=torch.get_default_dtype(), 
            device=TORCH_DEVICE
        )
        return self.lambda_p * torch.diag(diag)
#: </penalty_matrix>