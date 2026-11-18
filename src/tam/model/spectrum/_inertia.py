# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Load forecasting spectrum effects: thermal inertia and degree days.

Implements multi-timescale building thermal inertia via causal exponential smoothing
coupled with B-spline expansions (Goude et al. 2014) and C-infinity smooth softplus
heating and cooling degree-day hinges.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._penalty_coordinates import CoordinatePenaltyMixin
from ._temporal_utils import build_clamped_knots, causal_linear_recurrence, cox_de_boor as _cox_de_boor, eval_clamped_spline_with_linear_tails, from_btk, to_btd


def _parse_float_sequence(vals: Union[Sequence[float], str, float, int]) -> Tuple[float, ...]:
    """Helper to parse a sequence of floats from string, scalar, or iterable."""
    if isinstance(vals, str):
        clean = vals.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace(",", "|")
        return tuple(float(v.strip()) for v in clean.split("|") if v.strip())
    if isinstance(vals, (int, float)):
        return (float(vals),)
    return tuple(float(v) for v in vals)


class InertiaEffect(CoordinatePenaltyMixin, BaseEffect):
    r"""Multi-timescale temporal inertia effect for time series forecasting.

    Unifies different physical and operational inertia phenomena (thermal envelope,
    rotational, behavioral) into a single causal dynamic effect. For thermal inertia,
    follows Newton's law of cooling for structural building envelopes (Goude et al. 2014):

    .. math::

        \tilde{T}_t(\alpha) = \alpha T_t + (1 - \alpha) \tilde{T}_{t-1}(\alpha), \qquad \alpha \in (0, 1)

    Constructs causal exponentially smoothed trajectories across a dictionary of decay
    constants :math:`\{\alpha_1, \dots, \alpha_M\}` and evaluates each through a
    penalized B-spline basis. State is carried forward across fit-to-predict boundaries.

    Args:
        feature_name (str): Input feature column name.
        inertia_type (str): Type of inertia phenomenon ('thermal', 'rotational', 'behavioral').
            Defaults to ``'thermal'``.
        alphas (Sequence[float] or str): Exponential smoothing parameters :math:`\alpha_m \in (0, 1)`.
            Defaults to ``(0.05, 0.01, 0.002)``.
        k (int): Number of spline knots per decay scale. Defaults to 8.
        spline_degree (int): B-spline polynomial degree. Defaults to 3.
        penalty_order (int): Difference penalty order. Defaults to 2.
        lambda_p (float): Regularization weight. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    stateful = True

    def __init__(
        self,
        feature_name: str,
        inertia_type: str = "thermal",
        alphas: Union[Sequence[float], str, float] = (0.05, 0.01, 0.002),
        k: int = 8,
        spline_degree: int = 3,
        penalty_order: int = 2,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        self.inertia_type = str(inertia_type).lower().strip()
        super().__init__(feature_name, f"inertia_{self.inertia_type}", lambda_p, extrapolate)

        self.alphas = _parse_float_sequence(alphas)
        if len(self.alphas) == 0:
            raise ValueError(f"InertiaEffect({self.inertia_type}) requires at least one alpha value.")
        for a in self.alphas:
            if not (0.0 < a <= 1.0):
                raise ValueError(f"Inertia alpha must be in (0, 1], got {a}")

        self.k = max(2, int(k))
        self.spline_degree = int(spline_degree)
        self.penalty_order = int(penalty_order)

        self._n_spline_coeffs = self.k + self.spline_degree
        self._n_coeffs = len(self.alphas) * self._n_spline_coeffs
        # one smoothing weight per decay constant
        self._init_coordinates([f"alpha={a}" for a in self.alphas], [1.0] * len(self.alphas))
        self._x_bounds: Optional[Tuple[float, float]] = None
        self._check_temporal_extrapolate()

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the domain normalization bounds on the training tensor (all groups)."""
        self._x_bounds = None
        x_flat = x_col.flatten()
        if x_flat.numel() > 0:
            self._x_bounds = (float(x_flat.min().item()), float(x_flat.max().item()) + 1e-6)

    def summary_type(self) -> str:
        """Returns effect type name for summary."""
        return f"Inertia[{self.inertia_type}]"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return f"type={self.inertia_type}, alphas={self.alphas}, k={self.k}, deg={self.spline_degree}"

    def get_n_coeffs(self) -> int:
        """Returns total number of basis coefficients."""
        return self._n_coeffs

    def _build_knots(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Constructs canonical fixed B-spline knots on [-1.0, 1.0]."""
        return build_clamped_knots(self.k, self.spline_degree, device, dtype)

#: <inertia_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Builds causal thermal inertia B-spline feature columns with domain normalization.

        ``(B, T)`` in, ``(B, T, M * (k + degree))`` out, with ``M`` the number of decay constants. Each group is smoothed on its
        own; the smoothed value at the last step (``{"y": (B, M)}``) is the state to carry.
        """
        x3, shape, layout = to_btd(x_col, self.multi_input)
        if self._x_bounds is None:
            self.initialize(x_col)
        device, dtype = x3.device, x3.dtype
        x = x3[..., 0]
        B, T = x.shape
        if self._x_bounds is not None:
            x_low, x_high = self._x_bounds
            x_norm = 2.0 * (x - x_low) / max(1e-6, x_high - x_low) - 1.0
        else:
            x_norm = x

        alphas = torch.tensor(self.alphas, dtype=torch.float64, device=device)
        if state_in is not None and "y" in state_in:
            y_prev = state_in["y"].to(device=device, dtype=dtype)
            if tuple(y_prev.shape) != (B, len(self.alphas)):
                raise ValueError(f"state 'y' has shape {tuple(y_prev.shape)}, expected {(B, len(self.alphas))}: continue_state needs the same groups and features as training")
        else:
            y_prev = x_norm[:, :1].expand(B, len(self.alphas)).clone()      # no history: the smoothed value starts at the first input
        y = causal_linear_recurrence(x_norm.unsqueeze(-1).expand(B, T, len(self.alphas)), 1.0 - alphas, alphas, y_prev)

        knots = self._build_knots(device=device, dtype=dtype)
        smoothed_blocks: List[torch.Tensor] = []
        for m in range(len(self.alphas)):
            y_m = y[..., m].reshape(-1)                                   # (B * T,) smoothed input of decay constant m
            smoothed_blocks.append(eval_clamped_spline_with_linear_tails(y_m, knots, self.spline_degree))   # (B * T, k + degree)

        self._state_out = {"y": y[:, -1, :].detach().clone()}
        phi = torch.cat(smoothed_blocks, dim=-1).reshape(B, T, self._n_coeffs)
        return from_btk(phi, shape, layout)
#: </inertia_feature_map>

#: <inertia_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds block-diagonal second-difference penalty matrix."""
        n_sp = self._n_spline_coeffs
        I = torch.eye(n_sp, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        if self.penalty_order == 0:
            D = I
        else:
            D = torch.diff(I, n=self.penalty_order, dim=0)
        DtD = D.T @ D
        return torch.block_diag(*[lam * DtD for lam in self.penalty_coordinates()])
#: </inertia_penalty_matrix>



ThermalInertiaEffect = InertiaEffect
