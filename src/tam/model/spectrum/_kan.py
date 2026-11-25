# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Kolmogorov-Arnold Network (KAN) edge-spline effect.

Implements Kolmogorov-Arnold representation (Kolmogorov 1957; Liu et al. 2024):

.. math::

    f(x_1, \dots, x_d) = \sum_{q=1}^Q \Phi_q\left( \sum_{p=1}^d \psi_{q,p}(x_p) \right)

where inner edge functions :math:`\psi_{q,p}(x_p)` combine residual SiLU and
B-spline expansions, and outer functions :math:`\Phi_q` evaluate penalized
splines on normalized aggregation nodes :math:`z_q = \tanh(\sum_p \psi_{q,p}(x_p))`.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import cox_de_boor as _cox_de_boor, from_btk, to_btd


class KANEffect(BaseEffect):
    r"""Kolmogorov-Arnold Network (KAN) edge-spline effect.

    Decomposes high-dimensional multivariate interactions :math:`(x_1, \dots, x_d)` into
    additive compositions of learnable inner edge functions and outer B-splines,
    avoiding the :math:`\mathcal{O}(k^d)` exponential complexity of tensor splines.

    Args:
        feature_name (str): Primary input feature column.
        additional_features (Optional[List[str]]): Additional input feature columns.
        n_nodes (int): Number of inner aggregation nodes :math:`Q`. Defaults to 8.
        k_inner (int): Number of inner edge B-spline basis functions. Defaults to 6.
        k_outer (int): Number of outer B-spline basis functions per node. Defaults to 8.
        spline_degree (int): B-spline polynomial degree. Defaults to 3.
        penalty_order (int): Difference penalty order for outer splines. Defaults to 2.
        lambda_p (float): Regularization penalty strength. Defaults to 1.0.
        seed (int): Deterministic random seed for inner edge initialization. Defaults to 42.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        additional_features: Optional[List[str]] = None,
        n_nodes: int = 8,
        k_inner: int = 6,
        k_outer: int = 8,
        spline_degree: int = 3,
        penalty_order: int = 2,
        edge_pen: float = 0.1,
        lambda_p: float = 1.0,
        seed: int = 42,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "kan", lambda_p, extrapolate)
        self.n_nodes = int(n_nodes)
        self.k_inner = int(k_inner)
        self.k_outer = int(k_outer)
        self.spline_degree = int(spline_degree)
        self.penalty_order = int(penalty_order)
        self.edge_pen = float(edge_pen)
        self.seed = int(seed)

        self.additional_features = additional_features or []
        self.input_features = [feature_name] + self.additional_features

        self.n_in_features = len(self.input_features)
        self.n_outer_coeffs = self.k_outer

        # Per-feature standardization and inner knot spans
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self.knots_in: Optional[torch.Tensor] = None
        self.knots_out = torch.linspace(
            -1.2, 1.2, self.k_outer + self.spline_degree + 1, device=TORCH_DEVICE
        )

        # Inner parameters and node scale-stability statistics
        self.w_silu: Optional[torch.Tensor] = None
        self.c_spline: Optional[torch.Tensor] = None
        self._u_mean: Optional[torch.Tensor] = None
        self._u_std: Optional[torch.Tensor] = None

    @property
    def required_features(self) -> List[str]:
        """The input columns of the effect, in order."""
        return self.input_features

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"Q={self.n_nodes}, k_in={self.k_inner}, k_out={self.k_outer}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_nodes * self.n_outer_coeffs

    def _init_inner_parameters(self, x_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed)

        # drawn in float64 on the CPU, then cast: the same parameters on every device and dtype
        w = torch.randn(self.n_nodes, x_dim, generator=rng, dtype=torch.float64) * 0.25
        c = torch.randn(self.n_nodes, x_dim, self.k_inner, generator=rng, dtype=torch.float64) * 0.25

        self.w_silu = w.to(device=device, dtype=dtype)
        self.c_spline = c.to(device=device, dtype=dtype)

    def _get_inner_knots(self, x_std_in: torch.Tensor) -> torch.Tensor:
        """Builds per-feature inner knot grids of shape (d, k_inner + spline_degree + 1) from the training rows (once)."""
        d = x_std_in.shape[-1]
        if self.knots_in is None or self.knots_in.shape[0] != d:
            knot_rows = []
            n_knots = self.k_inner + self.spline_degree + 1
            for p in range(d):
                col_p = x_std_in[:, p]
                min_val = float(torch.min(col_p).item()) if col_p.numel() > 0 else -1.0
                max_val = float(torch.max(col_p).item()) if col_p.numel() > 0 else 1.0
                span = max(max_val - min_val, 1e-3)
                low = min_val - 0.1 * span
                high = max_val + 0.1 * span
                knot_rows.append(
                    torch.linspace(low, high, n_knots, device=x_std_in.device, dtype=x_std_in.dtype)
                )
            self.knots_in = torch.stack(knot_rows, dim=0)
        return self.knots_in.to(device=x_std_in.device, dtype=x_std_in.dtype)

    def _rows(self, x_col: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], str, int, int]:
        """The input as rows ``(B * T, d)`` plus what is needed to give the feature map its layout back."""
        multi = len(self.input_features) > 1 or (x_col.dim() == 2 and x_col.shape[-1] == 1)
        x3, shape, layout = to_btd(x_col, multi)
        B, T, d = x3.shape
        return x3.reshape(B * T, d), shape, layout, B, T

    def _standardize(self, x_in: torch.Tensor) -> torch.Tensor:
        """Per-feature standardization with the frozen training statistics, scaled to unit half-width."""
        x_mean = self._x_mean.to(device=x_in.device, dtype=x_in.dtype)
        x_std = self._x_std.to(device=x_in.device, dtype=x_in.dtype)
        return (x_in - x_mean) / (x_std * 1.7320508075688772)

    def _nodes(self, x_std_in: torch.Tensor) -> torch.Tensor:
        """Inner edge superpositions summed per node, ``(N, Q)`` (before the scale stabilization)."""
        d = x_std_in.shape[-1]
        w_silu = self.w_silu.to(device=x_std_in.device, dtype=x_std_in.dtype)
        c_spline = self.c_spline.to(device=x_std_in.device, dtype=x_std_in.dtype)
        knots_in = self._get_inner_knots(x_std_in)
        B_in = torch.cat([_cox_de_boor(x_std_in[:, p], knots_in[p], self.spline_degree) for p in range(d)], dim=-1)
        return (
            torch.nn.functional.silu(x_std_in) @ w_silu.T
            + B_in @ c_spline.reshape(self.n_nodes, -1).T
        )

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the inner parameters, the standardization, the inner knots and the node statistics on the training rows."""
        self._x_mean = None
        self._x_std = None
        self.knots_in = None
        self._u_mean = None
        self._u_std = None
        self.w_silu = None
        self.c_spline = None
        if x_col.numel() == 0:
            return
        x_in, _, _, _, _ = self._rows(x_col)
        d = x_in.shape[1]
        self._init_inner_parameters(d, x_in.dtype, x_in.device)
        self._x_mean = x_in.mean(dim=0, keepdim=True).detach()
        std_val = x_in.std(dim=0, keepdim=True, unbiased=False).detach()
        self._x_std = torch.where(std_val > 1e-12, std_val, torch.ones_like(std_val))
        u = self._nodes(self._standardize(x_in))
        self._u_mean = u.mean(dim=0, keepdim=True).detach()
        u_std_val = u.std(dim=0, keepdim=True, unbiased=False).detach()
        self._u_std = torch.where(u_std_val > 1e-12, u_std_val, torch.ones_like(u_std_val))

#: <kan_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Constructs the explicit Primal KAN design matrix.

        Args:
            x_col: ``(B, T, d)``, ``(B, T)`` for one input column, or ``(N, d)`` rows (a tensor-product margin).

        Returns:
            Design matrix ``(B, T, Q * k_outer)`` (``(N, Q * k_outer)`` for rows). KAN reads one row at a time: no state.
        """
        if self._x_mean is None:
            self.initialize(x_col)
        x_in, shape, layout, B, T = self._rows(x_col)
        if x_in.shape[1] != self.n_in_features:
            raise ValueError(f"kan expects {self.n_in_features} input column(s), got {x_in.shape[1]}")
        x_std_in = self._standardize(x_in)
        u = self._nodes(x_std_in)
        u_mean = self._u_mean.to(device=u.device, dtype=u.dtype)
        u_std = self._u_std.to(device=u.device, dtype=u.dtype)
        z = torch.tanh(0.28 * (u - u_mean) / u_std)  # (N_rows, Q)

        knots_out = self.knots_out.to(device=x_in.device, dtype=x_in.dtype)
        phi_kan = _cox_de_boor(z.reshape(-1), knots_out, self.spline_degree).reshape(z.shape[0], self.n_nodes * self.k_outer)
        return from_btk(phi_kan.reshape(B, T, self.n_nodes * self.k_outer), shape, layout)
#: </kan_feature_map>

#: <kan_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs block-diagonal second-difference Sobolev penalty weighted by inner-edge roughness:

        .. math::

            P_{\text{kan}} = \operatorname{blkdiag}_{q=1}^Q \left( (1 + \alpha_{\text{edge}} \mathcal{E}_q) D_{\text{order}}^\top D_{\text{order}} + \varepsilon_{\text{ridge}} I \right) \cdot \lambda_p
        """
        dtype = torch.get_default_dtype()
        I_sub = torch.eye(self.n_outer_coeffs, device=TORCH_DEVICE, dtype=dtype)
        if self.penalty_order == 0:
            D = I_sub
        else:
            D = torch.diff(I_sub, n=self.penalty_order, dim=0)

        D2_outer = D.T @ D
        eps_ridge = 1e-6 * I_sub

        if self.c_spline is None:
            self._init_inner_parameters(self.n_in_features, dtype, TORCH_DEVICE)
        c_sp = self.c_spline.to(device=TORCH_DEVICE, dtype=dtype)  # (Q, d, k_inner)

        if self.k_inner > 2:
            I_in = torch.eye(self.k_inner, device=TORCH_DEVICE, dtype=dtype)
            D_in = torch.diff(I_in, n=min(2, self.k_inner - 1), dim=0)
            # c_sp @ D_in.T has shape (Q, d, k_inner - 2)
            rough_in = torch.matmul(c_sp, D_in.T)
            e_q = torch.mean(torch.sum(rough_in ** 2, dim=-1), dim=-1)  # (Q,)
        else:
            e_q = torch.mean(torch.sum(c_sp ** 2, dim=-1), dim=-1)

        blocks = []
        for q in range(self.n_nodes):
            weight_q = 1.0 + self.edge_pen * float(e_q[q].item())
            blocks.append((weight_q * D2_outer + eps_ridge) * self.lambda_p)

        return torch.block_diag(*blocks)
#: </kan_penalty_matrix>
