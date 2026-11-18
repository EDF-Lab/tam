# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Fourier Neural Operator (FNO) spectrum effect.

Implements resolution-invariant spectral convolutions (Li et al. 2021; Kovachki et al. 2023):

.. math::

    u_m(t) = \operatorname{GELU}\Big( w_m v(t) + \mathcal{F}^{-1}\big( R_m(\omega) \cdot \mathcal{F}(v)(\omega) \big)(t) \Big)
"""

from typing import Any, Dict, Optional
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, causal_window_unfold, from_btk, tail_state, to_btd


class OperatorEffect(BaseEffect):
    r"""Fourier Neural Operator (FNO) spectral convolution effect.

    Maps continuous input trajectories to finite Primal channels via global Fourier
    spectral convolutions truncated at :math:`k_{\max}` modes, providing zero-shot
    discretization invariance across varying grid resolutions.

    Args:
        feature_name (str): Name of the input signal column.
        modes (int): Maximum number of Fourier modes retained. Defaults to 12.
        width (int): Number of spectral hidden channels. Defaults to 8.
        window (int): Length of the causal sliding window the spectral convolution reads. Defaults to 32.
        causal (bool): Whether each step reads only its causal window (True) or the whole pass at once (False). Defaults to True.
        seed (int): Deterministic random seed for operator parameter weights. Defaults to 42.
        lambda_p (float): Regularization penalty strength. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = False

    def __init__(
        self,
        feature_name: str,
        modes: int = 12,
        width: int = 8,
        window: int = 32,
        causal: bool = True,
        seed: int = 42,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "fno", lambda_p, extrapolate)
        self.modes = max(2, int(modes))
        self.width = max(1, int(width))
        self.window = max(4, int(window))
        self.causal = bool(causal)
        self.seed = int(seed)

        # Initialize deterministic spectral weights R and local bypass w
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)

        r_real = torch.randn(self.width, self.modes, generator=rng, dtype=torch.float64) * 0.15
        r_imag = torch.randn(self.width, self.modes, generator=rng, dtype=torch.float64) * 0.15
        self.R = torch.complex(r_real, r_imag)  # (width, modes)
        self.w_local = torch.randn(self.width, generator=rng, dtype=torch.float64) * 0.25

        self._n_hist = self.window - 1
        if self.causal:
            self._check_temporal_extrapolate()

    @property
    def stateful(self) -> bool:
        """The causal form carries the last ``window - 1`` samples; the resolution-invariant form reads the whole pass."""
        return self.causal

    def initialize(self, x_col: torch.Tensor) -> None:
        """The spectral weights come from the seed: nothing is learned from the data."""
        return None

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"modes={self.modes}, width={self.width}, W={self.window}, causal={self.causal}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.width

    def _mix(self, v_hat: torch.Tensor, m: int, k_modes: int) -> torch.Tensor:
        """The spectrum of ``v`` with its first ``k_modes`` modes multiplied by the weights of channel ``m`` and the others set to zero."""
        R = self.R[m : m + 1, :k_modes].to(device=v_hat.device, dtype=v_hat.dtype)
        out_hat = torch.zeros_like(v_hat)
        out_hat[..., :k_modes] = v_hat[..., :k_modes] * R
        return out_hat

#: <fno_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Constructs causal or resolution-invariant FNO spectral channels.

        Args:
            x_col: Input sequence tensor ``(B, T)``, ``(B, T, 1)`` or ``(T,)``; every group is read alone.
            state_in: ``{"hist": (B, window - 1, 1)}`` of the pass before (causal form).

        Returns:
            Design matrix :math:`\Phi \in \mathbb{R}^{B \times T \times \text{width}}`.
        """
        x3, shape, layout = to_btd(x_col, False)
        if x3.shape[-1] != 1:
            raise ValueError(f"fno reads one column, got {x3.shape[-1]}")
        v = x3[..., 0]  # (B, T)
        B, T = v.shape
        dtype = v.dtype
        w_loc = self.w_local.to(device=v.device, dtype=dtype)
        channels = []

        if self.causal:
            W = self.window
            x_hist = causal_history(x3, self._n_hist, state_in)
            self._state_out = {"hist": tail_state(x_hist, self._n_hist)}
            windows = causal_window_unfold(x_hist, W, t_len=T)[..., 0]  # (B, T, W)
            win_hat = torch.fft.rfft(windows, n=W, dim=-1)  # (B, T, W // 2 + 1)
            k_modes = min(self.modes, win_hat.shape[-1])
            for m in range(self.width):
                conv_m = torch.fft.irfft(self._mix(win_hat, m, k_modes), n=W, dim=-1)[..., -1]
                channels.append(torch.nn.functional.gelu(w_loc[m] * v + conv_m))
        else:
            # Non-causal spatial 1D FFT along the full domain of each group
            v_hat = torch.fft.rfft(v, n=T, dim=-1)  # (B, T // 2 + 1)
            k_modes = min(self.modes, v_hat.shape[-1])
            for m in range(self.width):
                conv_m = torch.fft.irfft(self._mix(v_hat, m, k_modes), n=T, dim=-1)
                channels.append(torch.nn.functional.gelu(w_loc[m] * v + conv_m))

        return from_btk(torch.stack(channels, dim=-1), shape, layout)
#: </fno_feature_map>

#: <fno_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs Sobolev operator smoothness penalty."""
        # Penalty weighted by spectral energy in high Fourier modes
        k_indices = torch.arange(1, self.modes + 1, dtype=torch.float64)
        spectral_roughness = torch.sum(
            (k_indices ** 2) * (torch.abs(self.R) ** 2), dim=-1
        ).to(device=TORCH_DEVICE, dtype=torch.get_default_dtype())

        diag_vals = (1.0 + spectral_roughness) * self.lambda_p
        return torch.diag(diag_vals)
#: </fno_penalty_matrix>
