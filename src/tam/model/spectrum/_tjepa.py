# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Temporal Joint-Embedding Predictive Architecture (T-JEPA) effect.

Extracts self-supervised, denoised latent representations from high-dimensional,
noisy exogenous covariate streams (LeCun 2022; Bardes et al. 2022; Assran et al. 2023):

.. math::

    s_x(t) = E_\vartheta(X_{t-w+1:t}), \qquad s_y(t) = \operatorname{stopgrad}(E_{\bar{\vartheta}}(X_{t+\Delta-w+1:t+\Delta}))

trained by predicting latent target representations with VICReg variance-invariance-covariance
regularization to prevent representation collapse, followed by an explicit Primal spline or
linear readout in StaticTAM.
"""

import math
from typing import Any, Dict, List, Optional
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, causal_window_unfold, cox_de_boor as _cox_de_boor, from_btk, tail_state, to_btd


class TJEPAEffect(BaseEffect):
    r"""Temporal Joint-Embedding Predictive Architecture (T-JEPA) latent covariate effect.

    Pre-trains a causal self-supervised context encoder :math:`E_\vartheta` on exogenous
    covariates using latent predictive loss and VICReg variance/covariance anti-collapse
    penalties, then projects frozen latents :math:`z_t = \tanh(E_\vartheta(X_t^{\text{ctx}}))`
    into TAM's Primal design matrix.

    Args:
        feature_name (str): Primary input feature column.
        additional_features (Optional[List[str]]): Additional exogenous feature columns.
        latent_dim (int): Dimensionality of the latent representation :math:`d`. Defaults to 4.
        window (int): Causal context window length :math:`w`. Defaults to 16.
        horizon (int): Future prediction step horizon :math:`\Delta`. Defaults to 4.
        readout (str): Readout type, either ``'linear'`` or ``'spline'``. Defaults to ``'spline'``.
        k_spline (int): Number of outer B-spline knots per latent coordinate. Defaults to 8.
        mode (str): Encoder fit, ``'svd'`` (closed-form predictive modes, deterministic; the default) or ``'vicreg'`` (self-supervised
            gradient training). Defaults to ``'svd'``.
        epochs (int): Number of self-supervised pre-training epochs; read only by ``mode='vicreg'`` (the optimizer settings are fixed). Defaults to 30.
        seed (int): Deterministic random seed. Defaults to 42.
        lambda_p (float): Regularization penalty strength. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = True
    stateful = True

    def __init__(
        self,
        feature_name: str,
        additional_features: Optional[List[str]] = None,
        latent_dim: int = 4,
        window: int = 16,
        horizon: int = 4,
        readout: str = "spline",
        k_spline: int = 8,
        epochs: int = 30,
        seed: int = 42,
        mode: str = "svd",
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "tjepa", lambda_p, extrapolate)
        self.latent_dim = max(1, int(latent_dim))
        self.window = max(2, int(window))
        self.horizon = max(1, int(horizon))
        self.readout = str(readout).lower()
        self.k_spline = max(4, int(k_spline))
        self.epochs = max(5, int(epochs))
        self.seed = int(seed)
        self.mode = str(mode).lower()
        if self.mode not in ("svd", "vicreg"):
            raise ValueError(f"tjepa mode must be 'svd' or 'vicreg', got '{mode}'")

        self.additional_features = additional_features or []
        self.input_features = [feature_name] + self.additional_features
        self.n_in_features = len(self.input_features)

        # Spline parameters
        self.spline_degree = 3
        self.n_outer_coeffs = self.k_spline
        self.knots_out = torch.linspace(
            -1.1, 1.1, self.k_spline + self.spline_degree + 1, device=TORCH_DEVICE
        )

        # Trained context encoder and training standardization
        self.encoder: Optional[torch.nn.Module] = None
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self._latent_res: Optional[torch.Tensor] = None     # (d,) residual of the latent predictor on the training windows
        self._n_hist = self.window - 1
        self._check_temporal_extrapolate()

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"d={self.latent_dim}, w={self.window}, readout={self.readout}, mode={self.mode}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        if self.readout == "linear":
            return self.latent_dim
        # Spline readout: k_spline coefficients per latent coordinate
        return self.latent_dim * self.n_outer_coeffs

    def _to_btd(self, x_col: torch.Tensor):
        """``(B, T, D)``; a two-dimensional input of one column is ``(B, T)``, of several columns (or one: ``(N, 1)``) rows ``(N, D)``."""
        multi = self.n_in_features > 1 or (x_col.dim() == 2 and x_col.shape[-1] == 1)
        return to_btd(x_col, multi)

    def _standardize(self, x3: torch.Tensor) -> torch.Tensor:
        return (x3 - self._x_mean.to(x3.device, x3.dtype)) / (self._x_std.to(x3.device, x3.dtype) + 1e-6)

    def initialize(self, x_col: torch.Tensor) -> None:
        """Standardizes the training tensor, then fits the T-JEPA context encoder on the windows of every group (never across two groups)."""
        self._x_mean = None
        self._x_std = None
        self.encoder = None
        self._latent_res = None
        x3, _, _ = self._to_btd(x_col)
        B, T, D = x3.shape
        if T < self.window + self.horizon:
            raise ValueError("tjepa needs at least window + horizon training rows per group")
        x_flat = x3.reshape(-1, D)
        self._x_mean = x_flat.mean(dim=0, keepdim=True).detach()
        std_val = x_flat.std(dim=0, keepdim=True, unbiased=False).detach()
        self._x_std = torch.where(std_val > 1e-6, std_val, torch.ones_like(std_val))
        x_std3 = self._standardize(x3)
        if self.mode == "svd":
            self._fit_tjepa_svd(x_std3)
        else:
            self._fit_tjepa_encoder(x_std3)

    def _fit_tjepa_svd(self, X: torch.Tensor) -> None:
        r"""Closed-form encoder on the standardized ``(B, T, D)`` tensor: the latent predictive modes of the context windows.

        With :math:`X_c` the centred context windows and :math:`Y_c` the centred target windows (the window :math:`\Delta` steps later), the
        encoder is :math:`W = V_r S_r^{-1} \sqrt{n}\, U_c` where :math:`X_c = U S V^\top` (rank :math:`r`) and :math:`U_c` are the left singular
        vectors of the cross-covariance :math:`U_r^\top Y_c`; the latent is :math:`\tanh(W^\top x + b)`. Windows never span two groups.
        """
        B, T, D = X.shape
        w = self.window
        delta = self.horizon
        d = self.latent_dim
        K = T - w - delta + 1
        all_windows = causal_window_unfold(X, w).reshape(B, T - w + 1, w * D)
        ctx_windows = all_windows[:, :K].reshape(-1, w * D)
        tgt_windows = all_windows[:, delta : delta + K].reshape(-1, w * D)
        n_pairs = ctx_windows.shape[0]
        feat_dim = w * D

        lin = torch.nn.Linear(feat_dim, d, bias=True, device=X.device, dtype=X.dtype)
        with torch.no_grad():
            if n_pairs < 16:
                # Too few windows: a fixed orthogonal projection
                with torch.random.fork_rng():
                    torch.manual_seed(self.seed)
                    torch.nn.init.orthogonal_(lin.weight)
                lin.bias.zero_()
                self.encoder = lin
                self._latent_res = torch.zeros(d, dtype=X.dtype, device=X.device)
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                return

            mu_c = ctx_windows.mean(dim=0, keepdim=True)
            mu_t = tgt_windows.mean(dim=0, keepdim=True)
            Xc = ctx_windows - mu_c
            Yc = tgt_windows - mu_t

            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            tol = max(float(S[0].item()) * 1e-6, 1e-9)
            valid = S > tol
            r = int(min(int(valid.sum().item()), max(d * 4, 24), feat_dim))
            r = max(r, 1)

            Z_white = U[:, :r]
            C_cross = Z_white.T @ Yc
            Uc, _, _ = torch.linalg.svd(C_cross, full_matrices=False)
            m_use = min(d, Uc.shape[1], r)
            scale_in = math.sqrt(float(n_pairs)) / S[:r].clamp_min(1e-6)
            W_cols = Vh[:r].T @ (scale_in.unsqueeze(1) * Uc[:, :m_use])

            if m_use < d:
                # The remaining directions are the leading principal modes
                extra = Vh[:d].T
                W = torch.zeros(feat_dim, d, dtype=X.dtype, device=X.device)
                W[:, :m_use] = W_cols
                for j in range(m_use, d):
                    col = extra[:, j % extra.shape[1]]
                    W[:, j] = col / col.norm().clamp_min(1e-6)
            else:
                W = W_cols[:, :d]

            lin.weight.copy_(W.T)
            lin.bias.copy_(-(mu_c.squeeze(0) @ W))

            # Relative error of a ridge latent predictor on each latent coordinate (penalty weights, as for the gradient encoder)
            z_ctx = torch.tanh(lin(ctx_windows))
            z_tgt = torch.tanh(lin(tgt_windows))
            reg = 1e-3 * float(n_pairs) * torch.eye(d, dtype=X.dtype, device=X.device)
            W_lat = torch.linalg.solve(z_ctx.T @ z_ctx + reg, z_ctx.T @ z_tgt)
            z_pred = z_ctx @ W_lat
            var_t = z_tgt.var(dim=0, unbiased=False).clamp_min(1e-8)
            self._latent_res = ((z_pred - z_tgt) ** 2).mean(dim=0) / var_t

        for p in lin.parameters():
            p.requires_grad_(False)
        self.encoder = lin

    def _fit_tjepa_encoder(self, X: torch.Tensor) -> None:
        """Pre-trains T-JEPA context encoder with VICReg self-supervised loss on the standardized ``(B, T, D)`` tensor."""
        B, T, D = X.shape
        w = self.window
        delta = self.horizon
        d = self.latent_dim

        K = T - w - delta + 1
        # Windows of each group alone: (B, T - w + 1, w * D); context window i is paired with the target window i + delta
        all_windows = causal_window_unfold(X, w).reshape(B, T - w + 1, w * D)
        ctx_windows = all_windows[:, :K].reshape(-1, w * D)
        tgt_windows = all_windows[:, delta : delta + K].reshape(-1, w * D)
        n_pairs = ctx_windows.shape[0]

        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            if K * B < 16:
                # Too few windows to train: a fixed linear projection
                self.encoder = torch.nn.Linear(w * D, d, device=X.device, dtype=X.dtype)
                self._latent_res = torch.zeros(d, dtype=X.dtype, device=X.device)
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                return

            enc = torch.nn.Sequential(
                torch.nn.Linear(w * D, 64, dtype=X.dtype, device=X.device),
                torch.nn.GELU(),
                torch.nn.Linear(64, d, dtype=X.dtype, device=X.device),
            )
            enc_ema = torch.nn.Sequential(
                torch.nn.Linear(w * D, 64, dtype=X.dtype, device=X.device),
                torch.nn.GELU(),
                torch.nn.Linear(64, d, dtype=X.dtype, device=X.device),
            )
            enc_ema.load_state_dict(enc.state_dict())

            pred = torch.nn.Sequential(
                torch.nn.Linear(d, 64, dtype=X.dtype, device=X.device),
                torch.nn.GELU(),
                torch.nn.Linear(64, d, dtype=X.dtype, device=X.device),
            )

            opt = torch.optim.AdamW(
                list(enc.parameters()) + list(pred.parameters()),
                lr=0.005,
                weight_decay=1e-4,
            )

            batch_size = 64
            for _ in range(self.epochs):
                perm = torch.randperm(n_pairs, device=X.device)
                for b in range(0, n_pairs, batch_size):
                    b_idx = perm[b : b + batch_size]
                    if len(b_idx) < 16:
                        continue

                    opt.zero_grad()
                    s_x = enc(ctx_windows[b_idx])
                    with torch.no_grad():
                        s_y = enc_ema(tgt_windows[b_idx])
                    s_y_hat = pred(s_x)
                    l_pred = torch.mean((s_y_hat - s_y) ** 2)

                    # VICReg anti-collapse penalties
                    s_x_cent = s_x - s_x.mean(dim=0, keepdim=True)
                    cov = (s_x_cent.T @ s_x_cent) / float(len(b_idx) - 1)
                    l_var = torch.mean(torch.relu(1.0 - torch.sqrt(torch.diagonal(cov) + 1e-4)))
                    cov_off = cov - torch.diag(torch.diagonal(cov))
                    l_cov = torch.sum(cov_off ** 2) / float(d)

                    loss = 2.0 * l_pred + 1.0 * l_var + 0.5 * l_cov
                    loss.backward()
                    opt.step()

                    # Target encoder EMA update
                    with torch.no_grad():
                        for p, p_ema in zip(enc.parameters(), enc_ema.parameters()):
                            p_ema.data.mul_(0.99).add_(p.data, alpha=0.01)

            # Relative error of the latent predictor on each latent coordinate: the more a coordinate is unpredictable, the more it is penalized
            with torch.no_grad():
                z_pred = pred(enc(ctx_windows))
                z_target = enc_ema(tgt_windows)
                var_target = z_target.var(dim=0, unbiased=False)
                self._latent_res = ((z_pred - z_target) ** 2).mean(dim=0) / (var_target + 1e-12)

        # Freeze fitted encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        self.encoder = enc

#: <tjepa_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Constructs causal T-JEPA latent feature map.

        Args:
            x_col: ``(B, T, D)`` (or ``(B, T)`` for one column, ``(T,)`` for one series).
            state_in: The last ``window - 1`` standardized rows of every group, ``{"hist": (B, w - 1, D)}``.

        Returns:
            Design matrix :math:`\Phi \in \mathbb{R}^{B \times T \times M}`.
        """
        x3, shape, layout = self._to_btd(x_col)
        if self.encoder is None:
            self.initialize(x_col)
        B, T, D = x3.shape
        if D != self.n_in_features:
            raise ValueError(f"tjepa expects {self.n_in_features} input column(s), got {D}")
        w = self.window
        x_std = self._standardize(x3)
        x_hist = causal_history(x_std, self._n_hist, state_in)
        self._state_out = {"hist": tail_state(x_hist, self._n_hist)}

        windows = causal_window_unfold(x_hist, w, t_len=T).reshape(B * T, w * D)     # (B * T, w * D), a window never spans two groups
        enc_param = next(self.encoder.parameters())
        with torch.no_grad():
            z = torch.tanh(self.encoder(windows.to(device=enc_param.device, dtype=enc_param.dtype))).to(device=x3.device, dtype=x3.dtype)

        if self.readout == "linear":
            phi = z
        else:
            # Spline readout: k_spline B-spline functions per latent coordinate
            knots_out = self.knots_out.to(device=x3.device, dtype=x3.dtype)
            phi = torch.cat([_cox_de_boor(z[:, j], knots_out, self.spline_degree) for j in range(self.latent_dim)], dim=-1)  # (B * T, d * k_spline)

        return from_btk(phi.reshape(B, T, self.get_n_coeffs()), shape, layout)
#: </tjepa_feature_map>

    def _latent_weights(self) -> torch.Tensor:
        """Penalty weight of every latent coordinate: ``1 + residual`` of the latent predictor (1 before the encoder is fitted)."""
        d = self.latent_dim
        dtype = torch.get_default_dtype()
        if self._latent_res is None:
            return torch.ones(d, device=TORCH_DEVICE, dtype=dtype)
        return 1.0 + self._latent_res.to(device=TORCH_DEVICE, dtype=dtype)

#: <tjepa_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs block-diagonal regularization penalty, block ``j`` weighted by ``1 + res_j``."""
        d = self.latent_dim
        weights = self._latent_weights()
        if self.readout == "linear":
            return torch.diag(self.lambda_p * weights)

        # Block-diagonal Sobolev roughness penalty
        I_sub = torch.eye(self.n_outer_coeffs, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        D = torch.diff(I_sub, n=2, dim=0)
        P_sub = (D.T @ D) * self.lambda_p
        return torch.block_diag(*[weights[j] * P_sub for j in range(d)])
#: </tjepa_penalty_matrix>
