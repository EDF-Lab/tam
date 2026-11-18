# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Singular Spectrum Analysis (SSA) effect.

Decomposes 1D time series into data-adaptive orthogonal empirical oscillation modes
(EOFs) via Hankel covariance matrix eigendecomposition (Broomhead & King 1986;
Golyandina et al. 2001):

.. math::

    \mathbf{H}_{i, j} = x_{i + j}, \quad \mathbf{C} = \frac{1}{K}\mathbf{H}\mathbf{H}^\top = \sum_{r=1}^L \sigma_r u_r u_r^\top

with causal sliding-window projection and state continuation:

.. math::

    \Phi_{\text{ssa}}(x)_{t, r} = u_r^\top x_{t-L+1:t}, \quad
    P_{\text{ssa}} = \operatorname{diag}\left( \frac{\sigma_1}{\sigma_r + \varepsilon} \right)_{r=1}^k \cdot \lambda_p
"""

from typing import Any, Dict, Optional
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, causal_window_unfold, from_btk, tail_state, to_btd


class SSAEffect(BaseEffect):
    r"""Singular Spectrum Analysis (SSA) empirical oscillation mode effect.

    Extracts anharmonic, amplitude-modulated, and data-adaptive empirical periodicities
    without requiring rigid harmonic or sinusoidal frequency assumptions.

    Args:
        feature_name (str): Name of the time series feature column.
        window (int): Embedding window length :math:`L`. Defaults to 48.
        k (int): Number of leading empirical orthogonal eigenvectors (EOFs) to retain. Defaults to 10.
        lambda_p (float): Base regularization strength. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = False
    stateful = True

    def __init__(
        self,
        feature_name: str,
        window: int = 48,
        k: int = 10,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "ssa", lambda_p, extrapolate)
        self.window = int(window)
        self.k = min(int(k), self.window)

        # Fitted EOF eigenvectors, singular values, and training mean
        self.eof_vectors: Optional[torch.Tensor] = None  # (window, k)
        self.singular_values: Optional[torch.Tensor] = None  # (k,)
        self._x_mean: Optional[torch.Tensor] = None

        self._n_hist = self.window - 1
        self._check_temporal_extrapolate()

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"L={self.window}, k={self.k}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.k

    def initialize(self, x_col: torch.Tensor) -> None:
        """Fits the centered Hankel EOF basis on the training tensor: every group has its own trajectory matrix."""
        self.eof_vectors = None
        self.singular_values = None
        self._x_mean = None
        x3, _, _ = to_btd(x_col, False)
        self._fit_eofs(x3[..., 0])

    def _fit_eofs(self, x: torch.Tensor) -> None:
        """Fits empirical orthogonal functions from the centered Hankel covariance matrix of the groups ``x`` ``(B, T)``."""
        B, T = x.shape
        L = self.window
        self._x_mean = x.mean().detach() if x.numel() > 0 else torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if T < L + 1:
            self.eof_vectors = torch.eye(L, self.k, device=x.device, dtype=x.dtype)
            self.singular_values = torch.ones(self.k, device=x.device, dtype=x.dtype)
            return

        x_cent = x - self._x_mean.to(device=x.device, dtype=x.dtype)
        K = T - L + 1
        H = x_cent.unfold(1, L, 1).reshape(B * K, L)       # the windows of each group, never across two groups: (B * K, L)
        C = (H.T @ H) / float(B * K)

        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals_desc = eigvals.flip(dims=[-1])
        eigvecs_desc = eigvecs.flip(dims=[-1])

        self.eof_vectors = eigvecs_desc[:, : self.k].contiguous()
        self.singular_values = torch.clamp(eigvals_desc[: self.k], min=1e-8).contiguous()

#: <ssa_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Constructs causal sliding-window projection onto fitted EOFs.

        Args:
            x_col: Input sequence tensor ``(B, T)`` (or ``(T,)``); every group is read alone.
            state_in: ``{"hist": (B, window - 1, 1)}``, the last centered samples of the pass before.

        Returns:
            Design matrix ``(B, T, k)``.
        """
        x3, shape, layout = to_btd(x_col, False)
        if x3.shape[-1] != 1:
            raise ValueError(f"ssa reads one column, got {x3.shape[-1]}")
        if self.eof_vectors is None:
            self.initialize(x_col)
        T = x3.shape[1]
        L = self.window

        U = self.eof_vectors.to(device=x3.device, dtype=x3.dtype)
        x_cent = x3 - self._x_mean.to(device=x3.device, dtype=x3.dtype)
        x_hist = causal_history(x_cent, self._n_hist, state_in)
        self._state_out = {"hist": tail_state(x_hist, self._n_hist)}
        windows = causal_window_unfold(x_hist, L, t_len=T)[..., 0]  # (B, T, L)
        return from_btk(windows @ U, shape, layout)  # (B, T, k)
#: </ssa_feature_map>

#: <ssa_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs spectral smoothness penalty:

        .. math::

            P_{\text{ssa}} = \operatorname{diag}\left( \frac{\sigma_1}{\sigma_r + 10^{-6}} \right)_{r=1}^k \cdot \lambda_p
        """
        if self.singular_values is None:
            weights = torch.ones(self.k, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        else:
            sig = self.singular_values.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype())
            sig_max = sig[0]
            weights = sig_max / (sig + 1e-6)

        return torch.diag(weights) * self.lambda_p
#: </ssa_penalty_matrix>
