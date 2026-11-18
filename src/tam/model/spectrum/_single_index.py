# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Single-Index Distributed Lag Smooth effect.

Implements single-index distributed-lag smoothing:

.. math::

    f_{\text{si}}(X_t) = \sum_{m=1}^{k} \theta_m B_m(X_t^\top \alpha)
    \quad \text{subject to} \quad \|\alpha\|_2 = 1,\; \alpha_1 \ge 0

learning a smooth index projection vector :math:`\alpha` composed with an
outer penalized B-spline (Ichimura 1993; Wood 2017).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import build_clamped_knots, causal_history, causal_window_unfold, cox_de_boor as _cox_de_boor, eval_clamped_spline_with_linear_tails, from_btk, tail_state, to_btd


class SingleIndexEffect(BaseEffect):
    r"""Penalized Single-Index Distributed Lag effect.

    Projects a multivariate block or lagged sliding-window trajectory :math:`X_t \in \mathbb{R}^L`
    onto a 1D scalar index :math:`u_t = X_t^\top \alpha` and evaluates a :math:`k`-knot cubic
    B-spline :math:`s(u_t)`.

    Args:
        feature_name (str): Primary input feature column.
        additional_features (Optional[List[str]]): Additional covariate columns forming :math:`X_t`.
        k (int): Number of spline knots. Defaults to 10.
        lag_window (Optional[int]): Number of causal lags :math:`[x_t, \dots, x_{t-L+1}]` to construct
            if univariate. Defaults to None.
        alpha (Optional[Sequence[float]]): Fixed or initial index weights. Defaults to None.
        alpha_pen (float): Roughness penalty weight for :math:`\alpha`. Defaults to 1.0.
        spline_degree (int): B-spline polynomial degree. Defaults to 3.
        penalty_order (int): Difference penalty order. Defaults to 2.
        lambda_p (float): Regularization weight. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        additional_features: Optional[List[str]] = None,
        k: int = 10,
        lag_window: Optional[int] = None,
        alpha: Optional[Union[Sequence[float], str]] = None,
        alpha_pen: float = 1.0,
        spline_degree: int = 3,
        penalty_order: int = 2,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "single_index", lambda_p, extrapolate)

        self.additional_features = additional_features or []
        self.input_features = [self.feature_name] + self.additional_features
        self.k = max(2, int(k))
        self.lag_window = int(lag_window) if lag_window is not None and int(lag_window) > 1 else None
        self.alpha_pen = float(alpha_pen)
        self.spline_degree = int(spline_degree)
        self.penalty_order = int(penalty_order)

        self._n_coeffs = self.k + self.spline_degree
        self.u_bounds: Optional[Tuple[float, float]] = None
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self._n_hist = (self.lag_window - 1) if self.lag_window is not None else 0
        self._user_alpha: bool = alpha is not None

        # Parse or initialize index weight vector alpha
        if alpha is not None:
            if isinstance(alpha, str):
                clean = alpha.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(",", "|")
                vals = [float(v.strip()) for v in clean.split("|") if v.strip()]
            else:
                vals = [float(v) for v in alpha]
            a_vec = np.array(vals, dtype=np.float64)
            norm = np.linalg.norm(a_vec)
            if norm > 1e-12:
                a_vec = a_vec / norm
            if len(a_vec) > 0 and a_vec[0] < 0:
                a_vec = -a_vec
            self.alpha = a_vec
        else:
            self.alpha = None

    @property
    def stateful(self) -> bool:
        """The distributed-lag form reads a causal window and carries it; the other forms read one row at a time."""
        return self.lag_window is not None

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the standardization, the index vector alpha and the bounds of the index on the training tensor (all groups)."""
        self.u_bounds = None
        self._x_mean = None
        self._x_std = None
        if not self._user_alpha:
            self.alpha = None
        if x_col.numel() == 0:
            return
        x3, _, _ = to_btd(x_col, len(self.input_features) > 1 or (x_col.dim() == 2 and x_col.shape[-1] == 1))
        d = x3.shape[-1]
        if len(self.input_features) > 1:
            x_2d = x3.reshape(-1, d)
            self._x_mean = x_2d.mean(dim=0, keepdim=True).detach()
            std_val = x_2d.std(dim=0, keepdim=True, unbiased=False).detach()
            self._x_std = torch.where(std_val > 1e-6, std_val, torch.ones_like(std_val))
            self._get_or_init_alpha(d, (x_2d - self._x_mean) / (self._x_std + 1e-6))
        elif self.lag_window is not None:
            self._get_or_init_alpha(self.lag_window)
        else:
            self._get_or_init_alpha(1)
        u, _ = self._index(x3, None)
        self.u_bounds = (float(u.min().item()), float(u.max().item()) + 1e-6)

    def summary_type(self) -> str:
        """Returns effect type name for summary."""
        return "SingleIndex"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        dim_str = f"L={self.lag_window}" if self.lag_window else f"dim={len(self.input_features)}"
        return f"{dim_str}, k={self.k}, deg={self.spline_degree}"

    def get_n_coeffs(self) -> int:
        """Returns number of basis coefficients."""
        return self._n_coeffs

    def _build_knots(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Constructs canonical fixed B-spline knots on [-1.0, 1.0]."""
        return build_clamped_knots(self.k, self.spline_degree, device, dtype)

    def _get_or_init_alpha(self, dim: int, x_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns index projection vector alpha, initializing via regularized SVD/decay if needed."""
        if self.alpha is not None and len(self.alpha) == dim:
            return torch.tensor(self.alpha, dtype=torch.get_default_dtype())

        if self.lag_window is not None:
            # Smooth decaying lag weights for distributed lags
            lags = np.arange(dim, dtype=np.float64)
            a = np.exp(-0.15 * lags)
        elif x_mat is not None and x_mat.dim() == 2 and x_mat.shape[0] >= dim and dim > 1:
            # Data-driven leading principal direction on standardized covariates, regularized by D2 penalty
            x_np = x_mat.detach().cpu().numpy().astype(np.float64)
            cov = (x_np.T @ x_np) / max(1, x_np.shape[0])
            if dim > 2 and self.alpha_pen > 0:
                d2 = np.diff(np.eye(dim), n=2, axis=0)
                reg_mat = cov + 1e-3 * self.alpha_pen * (d2.T @ d2)
            else:
                reg_mat = cov
            try:
                eigvals, eigvecs = np.linalg.eigh(reg_mat)
                a = eigvecs[:, -1]
            except np.linalg.LinAlgError:
                a = np.ones(dim, dtype=np.float64)
        else:
            a = np.ones(dim, dtype=np.float64)

        norm = np.linalg.norm(a)
        if norm > 1e-12:
            a = a / norm
        if len(a) > 0 and a[0] < 0:
            a = -a
        self.alpha = a
        return torch.tensor(self.alpha, dtype=torch.get_default_dtype())

    def _eval_normalized_spline(self, u: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        """Normalizes projected scalar index u to [-1, 1] using cached u_bounds and applies linear extrapolation."""
        u_flat = u.flatten()

        if self.u_bounds is not None:
            u_low, u_high = self.u_bounds
            u_norm = 2.0 * (u_flat - u_low) / max(1e-6, u_high - u_low) - 1.0
        else:
            u_norm = u_flat

        return eval_clamped_spline_with_linear_tails(u_norm, knots, self.spline_degree)

    def _index(self, x3: torch.Tensor, state_in: Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """The scalar index ``u`` of every step, ``(B, T)``, and the state to carry (the lag window only)."""
        B, T, d = x3.shape
        device, dtype = x3.device, x3.dtype
        if len(self.input_features) > 1:
            if d != len(self.input_features):
                raise ValueError(f"single-index expects {len(self.input_features)} input columns, got {d}")
            x_std = (x3 - self._x_mean.to(device, dtype)) / (self._x_std.to(device, dtype) + 1e-6)
            alpha = self._get_or_init_alpha(d).to(device=device, dtype=dtype)
            return torch.matmul(x_std, alpha), None
        if self.lag_window is not None:
            L = self.lag_window
            x_hist = causal_history(x3[..., :1], L - 1, state_in)
            windows = causal_window_unfold(x_hist, L, t_len=T)[..., 0].flip(-1)  # col 0 is x_t, col 1 is x_{t-1}
            alpha = self._get_or_init_alpha(L).to(device=device, dtype=dtype)
            return torch.matmul(windows, alpha), {"hist": tail_state(x_hist, L - 1)}
        alpha = self._get_or_init_alpha(1).to(device=device, dtype=dtype)
        return x3[..., 0] * alpha[0], None

#: <si_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Constructs causal single-index B-spline design matrix: ``(B, T, d)`` in, ``(B, T, k + degree)`` out."""
        x3, shape, layout = to_btd(x_col, len(self.input_features) > 1 or (x_col.dim() == 2 and x_col.shape[-1] == 1))
        if self.u_bounds is None:
            self.initialize(x_col)
        knots = self._build_knots(device=x3.device, dtype=x3.dtype)
        u, state = self._index(x3, state_in)
        phi = self._eval_normalized_spline(u, knots).reshape(u.shape[0], u.shape[1], self._n_coeffs)
        if state is not None:
            self._state_out = state
        return from_btk(phi, shape, layout)
#: </si_feature_map>

#: <si_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds second-difference penalty matrix for outer B-spline."""
        n_coeffs = self.get_n_coeffs()
        I = torch.eye(n_coeffs, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        if self.penalty_order == 0:
            D = I
        else:
            D = torch.diff(I, n=self.penalty_order, dim=0)
        return self.lambda_p * (D.T @ D)
#: </si_penalty_matrix>
