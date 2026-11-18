# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Sliding-Window Path Signature and Log-Signature effect for dynamical time series.

Extracts non-linear path dynamics, building thermal inertia, and asymmetric cross-channel
lead-lag causal features via truncated iterated integrals (Chen 1958; Lyons et al. 2007).
Supports Hoff's discrete Lead-Lag transform for rough quadratic variation
and Free Lie algebra compression via the Lyndon basis of the tensor logarithm.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, causal_window_unfold, from_btk, tail_state, to_btd
from ._signature_ops import (
    compute_witt_dimension,
    get_lyndon_words,
    lyndon_basis_indices,
)


class SignatureEffect(BaseEffect):
    r"""Sliding-window Path Signature effect.

    Maps a trajectory of covariates over a sliding window of length :math:`w` into
    a finite-dimensional linear coordinate system of iterated integrals truncated at
    level :math:`N`.

    Args:
        feature_name (str): Primary input feature column name.
        w (int): Sliding window length in rows/steps. Defaults to 9.
        N (int): Signature truncation order / level. Defaults to 3.
        lag (int): Delay in rows applied before the window end. Defaults to 0.
        time_augment (bool): Whether to append a normalized time channel :math:`\tau \in [0, 1]`.
            Defaults to True.
        log_sig (bool): Whether to compress the signature into the Free Lie algebra via the
            tensor logarithm and Lyndon word basis (eliminates shuffle collinearity).
            Defaults to False.
        lead_lag (bool): Whether to apply Hoff's discrete Lead-Lag transformation to capture
            cross-channel quadratic covariation and causal lead-lag geometry. Defaults to False.
        lambda_p (float): Base ridge regularization parameter. Defaults to 1.0.
        extrapolate (str): Extrapolation strategy outside training range. Defaults to ``'continue'``.
        additional_features (Optional[List[str]]): Additional covariate column names.
        k (Optional[int]): Total coefficient budget or level override.
    """

    stateful = True

    def __init__(
        self,
        feature_name: str,
        w: int = 9,
        N: int = 3,
        lag: int = 0,
        time_augment: bool = True,
        log_sig: bool = False,
        lead_lag: bool = False,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
        k: Optional[int] = None,
    ):
        """Initializes the SignatureEffect."""
        super().__init__(feature_name, "signature", lambda_p, extrapolate)
        self._check_temporal_extrapolate()

        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)

        self.w = max(2, int(w))
        self.lag = max(0, int(lag))
        self.time_augment = bool(time_augment)
        self.log_sig = bool(log_sig)
        self.lead_lag = bool(lead_lag)

        if k is not None and int(k) > 0:
            self.N = min(4, max(1, int(k)))
        else:
            self.N = min(4, max(1, int(N)))

        # Alphabet dimension D
        d_in = len(self.input_features)
        d_eff = 2 * d_in if self.lead_lag else d_in
        self.D = d_eff + 1 if self.time_augment else d_eff

        # Cache Lyndon basis indices if log-signature compression is enabled
        # When time_augment=True and D > 1, pure-time power indices (D-1, ..., D-1)
        # are deterministic constants across all windows and are dropped.
        if self.log_sig:
            self.lyndon_indices = lyndon_basis_indices(self.D, self.N)
            raw_witt = sum(compute_witt_dimension(self.D, lvl) for lvl in range(1, self.N + 1))
            self._n_coeffs = (raw_witt - 1) if (self.time_augment and self.D > 1) else raw_witt
        else:
            self.lyndon_indices = None
            if self.D == 1:
                self._n_coeffs = self.N
            elif self.time_augment:
                self._n_coeffs = sum(self.D ** lvl - 1 for lvl in range(1, self.N + 1))
            else:
                self._n_coeffs = sum(self.D ** lvl for lvl in range(1, self.N + 1))

        self._delta_scale: Optional[torch.Tensor] = None
        self._n_hist = self.w - 1 + self.lag

    def _windows(self, x3: torch.Tensor, state_in: Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal windows ``(B, T, w, d)`` ending ``lag`` steps before each step, and the history to carry (``(B, w - 1 + lag, d)``)."""
        T = x3.shape[1]
        x_hist = causal_history(x3, self._n_hist, state_in)
        windows = causal_window_unfold(x_hist[:, : x_hist.shape[1] - self.lag], self.w, t_len=T)
        return windows, tail_state(x_hist, self._n_hist)

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the increment scale on the training windows of every group (windows never cross two groups)."""
        self._delta_scale = None
        if x_col.numel() == 0:
            return
        x3, _, _ = to_btd(x_col, self.multi_input)
        windows, _ = self._windows(x3, None)
        d_in = windows.shape[-1]
        raw_diffs = windows[:, :, 1:, :] - windows[:, :, :-1, :]
        std_diff = raw_diffs.reshape(-1, d_in).std(dim=0, unbiased=False).detach()
        self._delta_scale = torch.where(std_diff > 1e-6, std_diff * math.sqrt(float(self.w)), torch.ones_like(std_diff))

    def get_n_coeffs(self) -> int:
        """Returns the total number of non-constant signature basis coefficients."""
        return self._n_coeffs

    def summary_type(self) -> str:
        """Returns the summary type."""
        return "LogSignature" if self.log_sig else "Signature"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return (
            f"w={self.w}, N={self.N}, D={self.D}, lag={self.lag}, "
            f"time_aug={self.time_augment}, log_sig={self.log_sig}, lead_lag={self.lead_lag}"
        )

    def _signatures_of_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """Signature (or log-signature) of every standardized window: ``(M, w, d_in) -> (M, n_coeffs)``."""
        M, w, d_in = windows.shape
        N = self.N
        D = self.D

        if M == 0:
            return torch.zeros(0, self._n_coeffs, dtype=windows.dtype, device=windows.device)

        # 3. Hoff's Lead-Lag transform
        if self.lead_lag:
            p_len = 2 * w - 1
            ll_windows = torch.zeros(
                M, p_len, 2 * d_in, dtype=windows.dtype, device=windows.device
            )
            ll_windows[:, 0, :d_in] = windows[:, 0]
            ll_windows[:, 0, d_in:] = windows[:, 0]
            for k in range(w - 1):
                ll_windows[:, 2 * k + 1, :d_in] = windows[:, k]
                ll_windows[:, 2 * k + 1, d_in:] = windows[:, k + 1]
                ll_windows[:, 2 * k + 2, :d_in] = windows[:, k + 1]
                ll_windows[:, 2 * k + 2, d_in:] = windows[:, k + 1]
            path_windows = ll_windows
        else:
            path_windows = windows
            p_len = w

        # 4. Time augmentation
        if self.time_augment:
            t_coord = torch.linspace(
                0.0, 1.0, p_len, dtype=windows.dtype, device=windows.device
            ).unsqueeze(0).unsqueeze(-1).expand(M, p_len, 1)
            aug_windows = torch.cat([path_windows, t_coord], dim=-1)
        else:
            aug_windows = path_windows

        # 5. Translation invariance: subtract start point
        win_trans = aug_windows - aug_windows[:, 0:1, :]
        deltas = win_trans[:, 1:, :] - win_trans[:, :-1, :]
        n_steps = p_len - 1

        # 6. Vectorized Chen product accumulation across all T windows
        S1 = torch.zeros(M, D, dtype=windows.dtype, device=windows.device)
        S2 = torch.zeros(M, D, D, dtype=windows.dtype, device=windows.device) if N >= 2 else None
        S3 = torch.zeros(M, D, D, D, dtype=windows.dtype, device=windows.device) if N >= 3 else None
        S4 = torch.zeros(M, D, D, D, D, dtype=windows.dtype, device=windows.device) if N >= 4 else None

        for step in range(n_steps):
            d = deltas[:, step, :]
            d2 = torch.einsum("ti,tj->tij", d, d) if N >= 2 else None
            d3 = torch.einsum("tij,tk->tijk", d2, d) if N >= 3 else None
            d4 = torch.einsum("tijk,tl->tijkl", d3, d) if N >= 4 else None

            if N >= 4:
                S4 = (
                    S4
                    + torch.einsum("tijk,tl->tijkl", S3, d)
                    + 0.5 * torch.einsum("tij,tkl->tijkl", S2, d2)
                    + (1.0 / 6.0) * torch.einsum("ti,tjkl->tijkl", S1, d3)
                    + (1.0 / 24.0) * d4
                )
            if N >= 3:
                S3 = S3 + torch.einsum("tij,tk->tijk", S2, d) + 0.5 * torch.einsum(
                    "ti,tjk->tijk", S1, d2
                ) + (1.0 / 6.0) * d3
            if N >= 2:
                S2 = S2 + torch.einsum("ti,tj->tij", S1, d) + 0.5 * d2
            S1 = S1 + d

        drop_pure_time = self.time_augment and D > 1

        if not self.log_sig:
            s_list = [S1[:, :-1] if drop_pure_time else S1]
            if N >= 2:
                s2_flat = S2.reshape(M, -1)
                s_list.append(s2_flat[:, :-1] if drop_pure_time else s2_flat)
            if N >= 3:
                s3_flat = S3.reshape(M, -1)
                s_list.append(s3_flat[:, :-1] if drop_pure_time else s3_flat)
            if N >= 4:
                s4_flat = S4.reshape(M, -1)
                s_list.append(s4_flat[:, :-1] if drop_pure_time else s4_flat)
            return torch.cat(s_list, dim=-1)

        # Log-Signature via tensor logarithm:
        L1 = S1[:, :-1] if drop_pure_time else S1
        log_parts = [L1]

        if N >= 2:
            L2 = S2 - 0.5 * torch.einsum("ti,tj->tij", S1, S1)
            # Level 2 Lyndon words are (i, j) with i < j
            words_k2 = get_lyndon_words(D, 2)
            if words_k2:
                l2_cols = [L2[:, w[0], w[1]: w[1] + 1] for w in words_k2]
                log_parts.append(torch.cat(l2_cols, dim=-1))

        if N >= 3:
            term_sym = torch.einsum("ti,tjk->tijk", S1, S2) + torch.einsum("tij,tk->tijk", S2, S1)
            cube = torch.einsum("ti,tj,tk->tijk", S1, S1, S1)
            L3 = S3 - 0.5 * term_sym + (1.0 / 3.0) * cube
            words_k3 = get_lyndon_words(D, 3)
            if words_k3:
                l3_cols = [L3[:, w[0], w[1], w[2]: w[2] + 1] for w in words_k3]
                log_parts.append(torch.cat(l3_cols, dim=-1))

        if N >= 4:
            s1_s3 = torch.einsum("ti,tjkl->tijkl", S1, S3)
            s2_s2 = torch.einsum("tij,tkl->tijkl", S2, S2)
            s3_s1 = torch.einsum("tijk,tl->tijkl", S3, S1)
            term_quad = s1_s3 + s2_s2 + s3_s1

            s1_s1_s2 = torch.einsum("ti,tj,tkl->tijkl", S1, S1, S2)
            s1_s2_s1 = torch.einsum("ti,tjk,tl->tijkl", S1, S2, S1)
            s2_s1_s1 = torch.einsum("tij,tk,tl->tijkl", S2, S1, S1)
            term_cube = s1_s1_s2 + s1_s2_s1 + s2_s1_s1

            s1_quad = torch.einsum("ti,tj,tk,tl->tijkl", S1, S1, S1, S1)

            L4 = S4 - 0.5 * term_quad + (1.0 / 3.0) * term_cube - 0.25 * s1_quad
            words_k4 = get_lyndon_words(D, 4)
            if words_k4:
                l4_cols = [L4[:, w[0], w[1], w[2], w[3]: w[3] + 1] for w in words_k4]
                log_parts.append(torch.cat(l4_cols, dim=-1))

        return torch.cat(log_parts, dim=-1)

#: <sig_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Builds explicit causal path signature Primal basis matrix; ``(B, T)`` or ``(B, T, d)`` in, ``(B, T, K)`` out."""
        x3, shape, layout = to_btd(x_col, self.multi_input)
        if x3.shape[-1] != len(self.input_features):
            raise ValueError(f"signature expects {len(self.input_features)} input column(s), got {x3.shape[-1]}")
        if self._delta_scale is None:
            self.initialize(x_col)
        B, T, d_in = x3.shape
        windows, self_state = self._windows(x3, state_in)
        d_scale = self._delta_scale.to(device=x3.device, dtype=x3.dtype).view(1, 1, 1, d_in)
        windows = windows / (d_scale + 1e-8)
        phi = self._signatures_of_windows(windows.reshape(B * T, self.w, d_in)).reshape(B, T, self._n_coeffs)
        self._state_out = {"hist": self_state}
        return from_btk(phi, shape, layout)
#: </sig_feature_map>

#: <sig_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds strictly positive-definite level-dependent factorial Ridge penalty matrix.

        .. math::

            \lambda_k = \lambda \cdot (k!)^2

        Returns:
            torch.Tensor: Diagonal penalty matrix :math:`P_{\text{sig}}` of shape ``(n_coeffs, n_coeffs)``.
        """
        drop_pure_time = self.time_augment and self.D > 1
        weights = []
        for k in range(1, self.N + 1):
            level_fact = float(math.factorial(k)) ** 2
            if self.log_sig:
                witt_k = compute_witt_dimension(self.D, k)
                level_dim = (witt_k - 1) if (k == 1 and drop_pure_time) else witt_k
            else:
                if self.D == 1:
                    level_dim = 1
                elif drop_pure_time:
                    level_dim = self.D ** k - 1
                else:
                    level_dim = self.D ** k

            weights.extend([level_fact] * level_dim)

        diag = self.lambda_p * torch.tensor(
            weights, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
        )
        return torch.diag(diag)
#: </sig_penalty_matrix>
