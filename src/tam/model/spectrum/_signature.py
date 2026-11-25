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


def _exp_segment(d: torch.Tensor, n_lvl: int) -> tuple:
    r"""Evaluates truncated segment exponential :math:`\exp_\otimes(d)` up to level ``n_lvl`` (<= 4) without einsum."""
    s1 = d
    if n_lvl == 1:
        return (s1,)
    s2 = 0.5 * (d.unsqueeze(-1) * d.unsqueeze(-2))
    if n_lvl == 2:
        return (s1, s2)
    s3 = (1.0 / 3.0) * (s2.unsqueeze(-1) * d.unsqueeze(-2).unsqueeze(-3))
    if n_lvl == 3:
        return (s1, s2, s3)
    s4 = 0.25 * (s3.unsqueeze(-1) * d.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4))
    return (s1, s2, s3, s4)


def _chen_product(a: tuple, b: tuple, n_lvl: int) -> tuple:
    r"""Truncated Chen tensor product :math:`A \otimes B` over trailing tensor dimensions via flat broadcasting."""
    a1 = a[0]
    b1 = b[0]
    c1 = a1 + b1
    if n_lvl == 1:
        return (c1,)
    a2, b2 = a[1], b[1]
    c2 = a2 + a1.unsqueeze(-1) * b1.unsqueeze(-2) + b2
    if n_lvl == 2:
        return (c1, c2)
    a3, b3 = a[2], b[2]
    c3 = (
        a3
        + a2.unsqueeze(-1) * b1.unsqueeze(-2).unsqueeze(-3)
        + a1.unsqueeze(-1).unsqueeze(-1) * b2.unsqueeze(-3)
        + b3
    )
    if n_lvl == 3:
        return (c1, c2, c3)
    a4, b4 = a[3], b[3]
    c4 = (
        a4
        + a3.unsqueeze(-1) * b1.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4)
        + a2.unsqueeze(-1).unsqueeze(-1) * b2.unsqueeze(-3).unsqueeze(-4)
        + a1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * b3.unsqueeze(-4)
        + b4
    )
    return (c1, c2, c3, c4)


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
            self._lyndon_idx_k2 = tuple(zip(*get_lyndon_words(self.D, 2))) if self.N >= 2 else ()
            self._lyndon_idx_k3 = tuple(zip(*get_lyndon_words(self.D, 3))) if self.N >= 3 else ()
            self._lyndon_idx_k4 = tuple(zip(*get_lyndon_words(self.D, 4))) if self.N >= 4 else ()
        else:
            self.lyndon_indices = None
            self._lyndon_idx_k2 = ()
            self._lyndon_idx_k3 = ()
            self._lyndon_idx_k4 = ()
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

    def _format_signature(self, s_tuple: tuple, m_rows: int) -> torch.Tensor:
        """Converts graded signature levels ``S = (S1, ...)`` of shape ``(M, D, ...)`` into output feature columns."""
        N = self.N
        D = self.D
        S1 = s_tuple[0]
        S2 = s_tuple[1] if N >= 2 else None
        S3 = s_tuple[2] if N >= 3 else None
        S4 = s_tuple[3] if N >= 4 else None
        drop_pure_time = self.time_augment and D > 1

        if not self.log_sig:
            s_list = [S1[:, :-1] if drop_pure_time else S1]
            if N >= 2:
                s2_flat = S2.reshape(m_rows, -1)
                s_list.append(s2_flat[:, :-1] if drop_pure_time else s2_flat)
            if N >= 3:
                s3_flat = S3.reshape(m_rows, -1)
                s_list.append(s3_flat[:, :-1] if drop_pure_time else s3_flat)
            if N >= 4:
                s4_flat = S4.reshape(m_rows, -1)
                s_list.append(s4_flat[:, :-1] if drop_pure_time else s4_flat)
            return torch.cat(s_list, dim=-1)

        L1 = S1[:, :-1] if drop_pure_time else S1
        log_parts = [L1]

        if N >= 2:
            L2 = S2 - 0.5 * (S1.unsqueeze(-1) * S1.unsqueeze(-2))
            if self._lyndon_idx_k2:
                i2, j2 = self._lyndon_idx_k2
                log_parts.append(L2[:, i2, j2])

        if N >= 3:
            s1_s2 = S1.unsqueeze(-1).unsqueeze(-1) * S2.unsqueeze(-3)
            s2_s1 = S2.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3)
            s1_2 = S1.unsqueeze(-1) * S1.unsqueeze(-2)
            cube = s1_2.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3)
            L3 = S3 - 0.5 * (s1_s2 + s2_s1) + (1.0 / 3.0) * cube
            if self._lyndon_idx_k3:
                i3, j3, k3 = self._lyndon_idx_k3
                log_parts.append(L3[:, i3, j3, k3])

        if N >= 4:
            s1_s3 = S1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * S3.unsqueeze(-4)
            s2_s2 = S2.unsqueeze(-1).unsqueeze(-1) * S2.unsqueeze(-3).unsqueeze(-4)
            s3_s1 = S3.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4)
            term_quad = s1_s3 + s2_s2 + s3_s1

            s1_s1 = S1.unsqueeze(-1) * S1.unsqueeze(-2)
            s1_s1_s2 = s1_s1.unsqueeze(-1).unsqueeze(-1) * S2.unsqueeze(-3).unsqueeze(-4)
            s1_s2_s1 = s1_s2.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4)
            s2_s1_s1 = s2_s1.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4)
            term_cube = s1_s1_s2 + s1_s2_s1 + s2_s1_s1

            s1_quad = cube.unsqueeze(-1) * S1.unsqueeze(-2).unsqueeze(-3).unsqueeze(-4)
            L4 = S4 - 0.5 * term_quad + (1.0 / 3.0) * term_cube - 0.25 * s1_quad
            if self._lyndon_idx_k4:
                i4, j4, k4, l4 = self._lyndon_idx_k4
                log_parts.append(L4[:, i4, j4, k4, l4])

        return torch.cat(log_parts, dim=-1)

    def _signatures_of_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """Signature (or log-signature) of every standardized window: ``(M, w, d_in) -> (M, n_coeffs)``."""
        M, w, d_in = windows.shape
        N = self.N

        if M == 0:
            return torch.zeros(0, self._n_coeffs, dtype=windows.dtype, device=windows.device)

        if self.lead_lag:
            p_len = 2 * w - 1
            ll_windows = torch.empty(M, p_len, 2 * d_in, dtype=windows.dtype, device=windows.device)
            ll_windows[:, 0, :d_in] = windows[:, 0]
            ll_windows[:, 0, d_in:] = windows[:, 0]
            ll_windows[:, 1::2, :d_in] = windows[:, :-1]
            ll_windows[:, 1::2, d_in:] = windows[:, 1:]
            ll_windows[:, 2::2, :d_in] = windows[:, 1:]
            ll_windows[:, 2::2, d_in:] = windows[:, 1:]
            path_windows = ll_windows
        else:
            path_windows = windows
            p_len = w

        if self.time_augment:
            t_coord = torch.linspace(
                0.0, 1.0, p_len, dtype=windows.dtype, device=windows.device
            ).view(1, p_len, 1).expand(M, p_len, 1)
            aug_windows = torch.cat([path_windows, t_coord], dim=-1)
        else:
            aug_windows = path_windows

        deltas = aug_windows[:, 1:, :] - aug_windows[:, :-1, :]
        n_steps = p_len - 1

        seg_all = _exp_segment(deltas, N)
        cur = tuple(lvl[:, 0] for lvl in seg_all)
        for step in range(1, n_steps):
            nxt = tuple(lvl[:, step] for lvl in seg_all)
            cur = _chen_product(cur, nxt, N)

        return self._format_signature(cur, M)

    def _signatures_sliding_btd(self, x_src: torch.Tensor) -> torch.Tensor:
        """Exact sliding-window signature along ``(B, T + w - 1, d_in)`` via Two-Stack Block Prefix-Suffix Chen products."""
        B, len_src, d_in = x_src.shape
        w = self.w
        L = w - 1
        T = len_src - L
        N = self.N
        device, dtype = x_src.device, x_src.dtype
        if T <= 0 or B == 0:
            return torch.zeros(B, max(T, 0), self._n_coeffs, dtype=dtype, device=device)

        dx = x_src[:, 1:, :] - x_src[:, :-1, :]  # (B, T + L - 1, d_in)
        n_edges = dx.shape[1]

        if self.lead_lag:
            n_sub = 2 * L
            dt = 1.0 / float(n_sub)
            zeros_dx = torch.zeros_like(dx)
            if self.time_augment:
                dt_col = torch.full((B, n_edges, 1), dt, dtype=dtype, device=device)
                d_lead = torch.cat([zeros_dx, dx, dt_col], dim=-1)
                d_lag = torch.cat([dx, zeros_dx, dt_col], dim=-1)
            else:
                d_lead = torch.cat([zeros_dx, dx], dim=-1)
                d_lag = torch.cat([dx, zeros_dx], dim=-1)
            edges = _chen_product(_exp_segment(d_lead, N), _exp_segment(d_lag, N), N)
        else:
            dt = 1.0 / float(L)
            if self.time_augment:
                dt_col = torch.full((B, n_edges, 1), dt, dtype=dtype, device=device)
                d_step = torch.cat([dx, dt_col], dim=-1)
            else:
                d_step = dx
            edges = _exp_segment(d_step, N)

        if L == 1:
            s_win = tuple(lvl[:, :T] for lvl in edges)
        else:
            n_blocks = (n_edges + L - 1) // L
            n_pad = n_blocks * L
            pad_edges = []
            for lvl in edges:
                if n_pad > n_edges:
                    pad_tail = torch.zeros((B, n_pad - n_edges, *lvl.shape[2:]), dtype=dtype, device=device)
                    lvl_p = torch.cat([lvl, pad_tail], dim=1)
                else:
                    lvl_p = lvl
                pad_edges.append(lvl_p.view(B, n_blocks, L, *lvl.shape[2:]))
            E_blk = tuple(pad_edges)

            pref_steps: List[List[torch.Tensor]] = [[] for _ in range(N)]
            suff_steps: List[Optional[List[torch.Tensor]]] = [[None] * L for _ in range(N)]

            p_cur = tuple(lvl[:, :, 0] for lvl in E_blk)
            for m_idx in range(N):
                pref_steps[m_idx].append(p_cur[m_idx])
            for k in range(1, L):
                p_cur = _chen_product(p_cur, tuple(lvl[:, :, k] for lvl in E_blk), N)
                for m_idx in range(N):
                    pref_steps[m_idx].append(p_cur[m_idx])

            r_cur = tuple(lvl[:, :, L - 1] for lvl in E_blk)
            for m_idx in range(N):
                suff_steps[m_idx][L - 1] = r_cur[m_idx]
            for k in range(L - 2, -1, -1):
                r_cur = _chen_product(tuple(lvl[:, :, k] for lvl in E_blk), r_cur, N)
                for m_idx in range(N):
                    suff_steps[m_idx][k] = r_cur[m_idx]

            P_flat = tuple(torch.stack(pref_steps[m_idx], dim=2).reshape(B, n_pad, *E_blk[m_idx].shape[3:]) for m_idx in range(N))
            R_flat = tuple(torch.stack(suff_steps[m_idx], dim=2).reshape(B, n_pad, *E_blk[m_idx].shape[3:]) for m_idx in range(N))

            s_win_list = [lvl[:, :T].clone() for lvl in R_flat]
            idx_t = torch.arange(T, device=device)
            non_aligned = (idx_t % L) != 0
            if bool(non_aligned.any()):
                na_t = idx_t[non_aligned]
                left = tuple(lvl[:, na_t] for lvl in R_flat)
                right = tuple(lvl[:, na_t + (L - 1)] for lvl in P_flat)
                comb = _chen_product(left, right, N)
                for m_idx in range(N):
                    s_win_list[m_idx][:, na_t] = comb[m_idx]
            s_win = tuple(s_win_list)

        s_flat = tuple(lvl.reshape(B * T, *lvl.shape[2:]) for lvl in s_win)
        return self._format_signature(s_flat, B * T).reshape(B, T, self._n_coeffs)

#: <sig_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Builds explicit causal path signature Primal basis matrix; ``(B, T)`` or ``(B, T, d)`` in, ``(B, T, K)`` out."""
        x3, shape, layout = to_btd(x_col, self.multi_input)
        if x3.shape[-1] != len(self.input_features):
            raise ValueError(f"signature expects {len(self.input_features)} input column(s), got {x3.shape[-1]}")
        if self._delta_scale is None:
            self.initialize(x_col)
        T = x3.shape[1]
        x_hist = causal_history(x3, self._n_hist, state_in)
        self._state_out = {"hist": tail_state(x_hist, self._n_hist)}
        if T == 0:
            return from_btk(torch.zeros(x3.shape[0], 0, self._n_coeffs, dtype=x3.dtype, device=x3.device), shape, layout)
        d_scale = self._delta_scale.to(device=x3.device, dtype=x3.dtype).view(1, 1, x3.shape[-1])
        x_src = x_hist[:, : x_hist.shape[1] - self.lag, :] / (d_scale + 1e-8)
        phi = self._signatures_sliding_btd(x_src)
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
