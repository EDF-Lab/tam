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


class DegreeDayEffect(BaseEffect):
    r"""Smooth Softplus Heating and Cooling Degree-Day Hinge effect.

    Computes smooth :math:`C^\infty` degree-day hinge functions:

    .. math::

        \operatorname{HDD}_\beta(T; u_m) = \frac{1}{\beta} \ln\!\Big(1 + \exp\big(\beta (u_m - T)\big)\Big)
        \operatorname{CDD}_\beta(T; v_j) = \frac{1}{\beta} \ln\!\Big(1 + \exp\big(\beta (T - v_j)\big)\Big)

    with optional quadratic saturation terms for extreme cold/heat regimes.

    Args:
        feature_name (str): Input temperature column name.
        heat_thresholds (Sequence[float] or str): Heating reference thresholds :math:`u_m`.
            Defaults to ``(-0.2, 0.0)``.
        cool_thresholds (Sequence[float] or str): Cooling reference thresholds :math:`v_j`.
            Defaults to ``(0.3, 0.5)``.
        sharpness (float): Softplus sharpness parameter :math:`\beta > 0`. Defaults to 10.0.
        quad (bool): Whether to append saturated quadratic terms :math:`\operatorname{HDD}^2, \operatorname{CDD}^2`.
            Defaults to True.
        lambda_p (float): Regularization weight. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
    """

    def __init__(
        self,
        feature_name: str,
        heat_thresholds: Union[Sequence[float], str, float] = (-0.2, 0.0),
        cool_thresholds: Union[Sequence[float], str, float] = (0.3, 0.5),
        sharpness: float = 10.0,
        quad: bool = True,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "degree_day", lambda_p, extrapolate)

        # ascending: the penalty couples the hinges of one family in the order of their thresholds
        self.heat_thresholds = tuple(sorted(_parse_float_sequence(heat_thresholds)))
        self.cool_thresholds = tuple(sorted(_parse_float_sequence(cool_thresholds)))
        self.sharpness = max(0.1, float(sharpness))
        self.quad = bool(quad)

        n_hinges = len(self.heat_thresholds) + len(self.cool_thresholds)
        self._n_coeffs = 2 * n_hinges if self.quad else n_hinges

    def summary_type(self) -> str:
        """Returns effect type name for summary."""
        return "DegreeDay"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return (
            f"heat={self.heat_thresholds}, cool={self.cool_thresholds}, "
            f"beta={self.sharpness}, quad={self.quad}"
        )

    def get_n_coeffs(self) -> int:
        """Returns total number of basis coefficients."""
        return self._n_coeffs

#: <dd_feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Builds smooth softplus degree day columns."""
        x_exp = x_col.unsqueeze(-1)
        cols: List[torch.Tensor] = []

        # Heating degree-day hinges: softplus(u - T, beta)
        for u in self.heat_thresholds:
            z = float(u) - x_exp
            hdd = F.softplus(z, beta=self.sharpness)
            cols.append(hdd)
            if self.quad:
                cols.append(hdd ** 2)

        # Cooling degree-day hinges: softplus(T - v, beta)
        for v in self.cool_thresholds:
            z = x_exp - float(v)
            cdd = F.softplus(z, beta=self.sharpness)
            cols.append(cdd)
            if self.quad:
                cols.append(cdd ** 2)

        return torch.cat(cols, dim=-1)
#: </dd_feature_map>

#: <dd_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds the penalty: a ridge with the quadratic columns weighted 4, plus a first-difference coupling of the hinges of one family.

        .. math::

            P = \lambda\,\bigl(\operatorname{diag}(w) + D_1^\top D_1|_{\text{heating hinges}} + D_1^\top D_1|_{\text{cooling hinges}}\bigr),
            \qquad w = 1 \text{ (hinge)},\; 4 \text{ (squared hinge)}

        The coupling acts on the linear hinges ordered by threshold, inside each family only; with one threshold per side there is none.
        """
        n_coeffs = self.get_n_coeffs()
        dtype = torch.get_default_dtype()
        weights = torch.ones(n_coeffs, device=TORCH_DEVICE, dtype=dtype)
        P = torch.zeros(n_coeffs, n_coeffs, device=TORCH_DEVICE, dtype=dtype)
        step = 2 if self.quad else 1
        position = 0
        for count in (len(self.heat_thresholds), len(self.cool_thresholds)):
            cols = [position + step * i for i in range(count)]
            if self.quad:
                for c in cols:
                    weights[c + 1] = 4.0
            if count > 1:
                D1 = torch.diff(torch.eye(count, device=TORCH_DEVICE, dtype=dtype), n=1, dim=0)
                C = D1.T @ D1
                idx = torch.tensor(cols, device=TORCH_DEVICE)
                P[idx.unsqueeze(1), idx.unsqueeze(0)] += C
            position += step * count
        P = P + torch.diag(weights)
        return self.lambda_p * P
#: </dd_penalty_matrix>


# Same class under the name of its formula token ``smooth_hinge``.
SmoothHingeEffect = DegreeDayEffect


class WarpedCalendarEffect(CoordinatePenaltyMixin, BaseEffect):
    r"""Phase-warped calendar Fourier effect for holidays and regime shifts.

    Note: has no effect when ``group_col`` is the column read by ``wcal`` (e.g. ``group_col='tod'``): that column is constant inside a group,
    and the columns varying through ``by`` are spanned by a day-type term on the same ``by`` column.
    Use it in a model not grouped by its input column.

    On public holidays, school breaks, and bridge days, the diurnal or annual load
    curve undergoes both vertical amplitude scaling and horizontal phase warping
    (e.g. delayed morning wake-up ramp). Parameterizes a diffeomorphic phase-warping map:

    .. math::

        \tilde{\tau}_t = \tau_t + \sum_{r=1}^K \delta_r z_t \sin(r \tau_t)

    Linearized around the base harmonics via first-order Taylor expansion:

    .. math::

        \Phi_{\text{wcal}}(\tau_t, z_t) = \Big[\, \Phi_{\text{base}}(\tau_t) \;\Big|\; z_t \odot \partial_\tau \Phi_{\text{base}}(\tau_t) \;\Big|\; z_t \odot \Phi_{\text{base}}(\tau_t) \,\Big]

    regularized by a block-diagonal Sobolev roughness penalty.

    Args:
        feature_name (str): Periodic time/phase column name (e.g. 'tod', 'toy').
        regime_feature (Optional[str]): Regime indicator column name (e.g. 'is_holiday', 'day_type_jf').
        period (float): Period length (e.g. 48.0 for half-hourly tod, 365.25 for toy).
            Defaults to 48.0.
        n_harmonics (int): Number of Fourier harmonics :math:`K`. Defaults to 6.
        sobolev_order (float): Sobolev penalty exponent :math:`s \ge 1`. Defaults to 2.0.
        gamma_warp (float): Penalty multiplier for horizontal phase-warping terms. Defaults to 1.0.
        gamma_amp (float): Penalty multiplier for vertical amplitude-scaling terms. Defaults to 1.0.
        lambda_p (float): Global regularization weight. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'periodic'``.
    """

    def __init__(
        self,
        feature_name: str,
        regime_feature: Optional[str] = None,
        period: float = 48.0,
        n_harmonics: int = 6,
        sobolev_order: float = 2.0,
        gamma_warp: float = 1.0,
        gamma_amp: float = 1.0,
        lambda_p: float = 1.0,
        extrapolate: str = "periodic",
    ):
        super().__init__(feature_name, "wcal", lambda_p, extrapolate)

        self.additional_features = [regime_feature] if regime_feature else []
        self.input_features = [self.feature_name] + self.additional_features
        self.regime_feature = regime_feature
        self.period = max(1e-4, float(period))
        self.n_harmonics = max(1, int(n_harmonics))
        self.sobolev_order = float(sobolev_order)
        self.gamma_warp = float(gamma_warp)
        self.gamma_amp = float(gamma_amp)

        self.has_regime = bool(self.additional_features)
        self._n_coeffs = 6 * self.n_harmonics if self.has_regime else 2 * self.n_harmonics
        if self.has_regime:
            self._init_coordinates(["base", "warp", "amp"], [1.0, self.gamma_warp, self.gamma_amp])
        else:
            self._init_coordinates(["base"], [1.0])

    def summary_type(self) -> str:
        """Returns effect type name for summary."""
        return "WarpedCalendar"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return (
            f"period={self.period}, K={self.n_harmonics}, "
            f"by='{self.regime_feature}', s={self.sobolev_order}"
        )

    def get_n_coeffs(self) -> int:
        """Returns total number of basis coefficients."""
        return self._n_coeffs

#: <wcal_feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Constructs base Fourier, phase-derivative warping, and amplitude interaction columns."""
        orig_shape = x_col.shape

        if self.has_regime and x_col.dim() >= 2 and x_col.shape[-1] >= 2:
            phase_in = x_col[..., 0]
            regime_in = x_col[..., 1:]
            batch_shape = orig_shape[:-1]
        else:
            phase_in = x_col
            regime_in = None
            batch_shape = orig_shape

        # The input is the phase normalized from [0, period] to [-1, 1] (fixed_ranges()): angular phase in [0, 2*pi)
        tau = (np.pi * (phase_in + 1.0)).unsqueeze(-1)
        k_vec = torch.arange(1, self.n_harmonics + 1, device=x_col.device, dtype=x_col.dtype)

        angles = tau * k_vec
        cos_part = torch.cos(angles)
        sin_part = torch.sin(angles)

        base_cols = torch.cat([cos_part, sin_part], dim=-1)

        if not self.has_regime or regime_in is None:
            return base_cols.reshape(*batch_shape, self._n_coeffs)

        d_cos = -k_vec * sin_part
        d_sin = k_vec * cos_part
        d_cols = torch.cat([d_cos, d_sin], dim=-1)

        # Diffeomorphic phase-modulation profile g(tau) = (tau - pi)/pi in [-1, 1)
        # prevents exact collinearity between phase-derivative warp_cols and amp_cols.
        phase_envelope = (tau - np.pi) / np.pi
        warp_cols = regime_in * phase_envelope * d_cols
        amp_cols = regime_in * base_cols

        full = torch.cat([base_cols, warp_cols, amp_cols], dim=-1)
        return full.reshape(*batch_shape, self._n_coeffs)
#: </wcal_feature_map>

#: <wcal_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Constructs block-diagonal Sobolev frequency penalty matrix."""
        k_vec = torch.arange(1, self.n_harmonics + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        sob_weights = (k_vec ** self.sobolev_order)
        p_base = torch.cat([sob_weights, sob_weights])

        coords = self.penalty_coordinates()
        if not self.has_regime:
            return torch.diag(coords[0] * p_base)

        lam_base, lam_warp, lam_amp = coords
        return torch.diag(torch.cat([lam_base * p_base, lam_warp * p_base, lam_amp * p_base]))
#: </wcal_penalty_matrix>


class MacroTrendEffect(BaseEffect):
    r"""Hodrick-Prescott structural macro-trend effect with natural boundary conditions.

    Solves the HP L2 trend filtering problem (Hodrick & Prescott 1997) in TAM's Primal space
    using natural cubic splines with exact second-difference penalty :math:`P = D_2^\top D_2`.
    Beyond the training range the trend continues along its boundary tangent (a linear continuation):
    the extrapolated level drifts linearly with the horizon.

    Args:
        feature_name (str): Input time/trend column name.
        k (int): Number of trend knots. Defaults to 20.
        lambda_p (float): HP smoothing penalty weight :math:`\lambda_{\text{HP}}`.
            Defaults to 1600.0 (standard quarterly/half-hourly macro benchmark).
        extrapolate (str): Extrapolation mode. Defaults to ``'linear'``.
    """

    def __init__(
        self,
        feature_name: str,
        k: int = 20,
        lambda_p: float = 1600.0,
        extrapolate: str = "linear",
    ):
        super().__init__(feature_name, "hp_trend", lambda_p, extrapolate)
        self.k = max(4, int(k))
        self.spline_degree = 3
        self._n_coeffs = self.k + self.spline_degree
        self._x_bounds: Optional[Tuple[float, float]] = None

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the range of the trend axis on the training tensor; a second call replaces it."""
        self._x_bounds = None
        x_flat = x_col.flatten()
        if x_flat.numel() > 0:
            self._x_bounds = (float(x_flat.min()), float(x_flat.max()) + 1e-6)

    def summary_type(self) -> str:
        """Returns effect type name for summary."""
        return "MacroTrend"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return f"k={self.k}, lambda_hp={self.lambda_p}, natural_extrap='linear'"

    def get_n_coeffs(self) -> int:
        """Returns total number of basis coefficients."""
        return self._n_coeffs

#: <hp_trend_feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Builds B-spline trend columns with linear extrapolation beyond bounds."""
        orig_shape = x_col.shape
        x_flat = x_col.flatten()
        device = x_col.device
        dtype = x_col.dtype

        knots = build_clamped_knots(self.k, self.spline_degree, device, dtype)

        if self._x_bounds is None:
            self.initialize(x_col)
        x_low, x_high = self._x_bounds
        x_norm = 2.0 * (x_flat - x_low) / max(1e-6, x_high - x_low) - 1.0

        phi = eval_clamped_spline_with_linear_tails(x_norm, knots, self.spline_degree)      # (N, k + degree)

        return phi.reshape(*orig_shape, self._n_coeffs)
#: </hp_trend_feature_map>

#: <hp_trend_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Constructs exact second-difference penalty D_2^T D_2."""
        p = self.get_n_coeffs()
        d2 = torch.zeros(p - 2, p, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        for i in range(p - 2):
            d2[i, i] = 1.0
            d2[i, i + 1] = -2.0
            d2[i, i + 2] = 1.0
        return self.lambda_p * (d2.T @ d2)
#: </hp_trend_penalty_matrix>


def _build_canonical_knots(
    k: int, degree: int, device: torch.device, dtype: torch.dtype, x_min: float = -1.0, x_max: float = 1.0
) -> torch.Tensor:
    """Helper to build clamped B-spline knots on [x_min, x_max]."""
    internal = torch.linspace(x_min, x_max, k + 1, device=device, dtype=dtype)[1:-1]
    t_left = torch.full((degree + 1,), x_min, device=device, dtype=dtype)
    t_right = torch.full((degree + 1,), x_max, device=device, dtype=dtype)
    return torch.cat([t_left, internal, t_right])
