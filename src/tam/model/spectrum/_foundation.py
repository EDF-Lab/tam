# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Universal foundation basis (`foundation`) and its internal components: a quantile histogram, a multi-rate filterbank and a
structural prior dictionary. The causal patch basis (`patch_basis`) lives in `_patch_basis.py` and is one block of the composite.

Every component is a weight-free, closed-form basis: no network is trained, the coefficients are solved by penalized weighted
least squares (P-WLS) like those of any other spectrum effect. The components read ``(B, T, d)`` and treat every group alone.
"""

from typing import Optional, List, Dict, Any, Sequence
import math
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._patch_basis import PatchBasisEffect
from ._temporal_utils import causal_history, causal_linear_recurrence, causal_window_unfold, from_btk, tail_state, to_btd


def _build_d2_penalty(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n <= 2:
        return torch.eye(n, device=device, dtype=dtype)
    D2 = torch.zeros((n - 2, n), device=device, dtype=dtype)
    for i in range(n - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2.mT @ D2 + 1e-6 * torch.eye(n, device=device, dtype=dtype)


def _carried(state_in: Optional[Dict[str, Any]], key: str, expected: tuple, device, dtype) -> Optional[torch.Tensor]:
    """The state ``key`` moved to the input's device and dtype; its shape must be the one of the pass that produced it."""
    if state_in is None or key not in state_in:
        return None
    s = state_in[key].to(device=device, dtype=dtype)
    if tuple(s.shape) != expected:
        raise ValueError(f"state '{key}' has shape {tuple(s.shape)}, expected {expected}: continue_state needs the same groups and features as training")
    return s


class QuantileHistEffect(BaseEffect):
    r"""
    Quantile soft-histogram basis (`quant_hist`).

    The current value and the values of a causal window are standardized by the window statistics and spread over fixed bins
    centred on the quantiles of a standard normal distribution (a triangular soft assignment). The columns are the average of the
    soft occupancy of the current value and of the window: a bounded description of where the series stands in its own recent
    distribution (quantile binning of a scaled window, as in Ansari et al. 2024).
    """

    stateful = True
    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        n_bins: int = 16,
        window: int = 24,
        lambda_p: float = 1e-2,
        extrapolate: str = "continue",
    ):
        super().__init__(
            feature_name=feature_name,
            effect_type="quant_hist",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.n_bins = max(int(n_bins), 4)
        self.window = max(int(window), 2)
        self.n_coeffs = self.n_bins

        # Inverse-Gaussian quantile bin centers in [-2.5, 2.5]
        probs = (torch.arange(1, self.n_bins + 1, dtype=torch.float64) - 0.5) / float(self.n_bins)
        centers = math.sqrt(2.0) * torch.erfinv(2.0 * probs - 1.0)
        self._centers = centers
        diffs = centers[1:] - centers[:-1]
        widths = torch.ones_like(centers)
        widths[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
        widths[0] = diffs[0]
        widths[-1] = diffs[-1]
        self._widths = torch.clamp(widths, min=0.15)
        self._state_out: Optional[Dict[str, Any]] = None
        self._check_temporal_extrapolate()

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"n_bins={self.n_bins}, window={self.window}"

    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Design block of the effect for the input ``x_col``; ``state_in`` carries the causal state of a previous chunk (see the class docstring)."""
        x3, shape, layout = to_btd(x_col, True)
        x1 = x3[..., :1]
        B, T, _ = x1.shape
        W = self.window
        dtype = x1.dtype
        device = x1.device

        x_hist = causal_history(x1, W - 1, state_in, key="hist")
        self._state_out = {"hist": tail_state(x_hist, W - 1)}
        windows = causal_window_unfold(x_hist, W, t_len=T)[..., 0]  # (B, T, W)
        x_in = x1[..., 0]
        mu_w = torch.mean(windows, dim=-1, keepdim=True)
        std_w = torch.sqrt(torch.mean((windows - mu_w) ** 2, dim=-1, keepdim=True) + 1e-6)
        # Current standardized observation + local window empirical distribution
        z_curr = (x_in.unsqueeze(-1) - mu_w) / std_w  # (B, T, 1)

        centers = self._centers.to(device=device, dtype=dtype).view(1, 1, -1)  # (1, 1, B_bins)
        widths = self._widths.to(device=device, dtype=dtype).view(1, 1, -1)
        # Soft triangular bin membership on current standardized value (50%) + window occupancy (50%)
        tri_curr = torch.clamp(1.0 - torch.abs(z_curr - centers) / widths, min=0.0)  # (B, T, B_bins)
        z_win = (windows - mu_w) / std_w  # (B, T, W)
        tri_win = torch.mean(
            torch.clamp(
                1.0 - torch.abs(z_win.unsqueeze(-1) - centers.unsqueeze(2)) / widths.unsqueeze(2),
                min=0.0,
            ),
            dim=2,
        )  # (B, T, B_bins)

        out = 0.5 * tri_curr + 0.5 * tri_win
        return from_btk(out, shape, layout)

    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        return self.lambda_p * _build_d2_penalty(self.n_bins, TORCH_DEVICE, torch.get_default_dtype())


class MultiRateFilterbankEffect(BaseEffect):
    r"""
    Multi-rate causal filterbank (`mrfb`).

    For every time scale :math:`s` an exponential moving average :math:`m_t = (1-a)\,m_{t-1} + a\,x_t` with
    :math:`a = 2/(s+1)`, the band :math:`x_t - m_t` (what the scale does not explain) and a bounded envelope of the band energy
    (the same recurrence on :math:`\tanh(\tfrac14 (x_t - m_t)^2)`). Each recurrence is solved along time in linear time and memory.
    """

    stateful = True
    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        scales: Sequence[int] = (1, 4, 16, 64),
        gamma_scale: float = 0.5,
        lambda_p: float = 1e-2,
        extrapolate: str = "continue",
    ):
        super().__init__(
            feature_name=feature_name,
            effect_type="mrfb",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.scales = [max(int(s), 1) for s in scales]
        self.gamma_scale = float(gamma_scale)
        # 3 channels per scale: causal EMA low-pass, causal multi-scale difference (band-pass), and envelope energy
        self.n_coeffs = 3 * len(self.scales)
        self._state_out: Optional[Dict[str, Any]] = None
        self._check_temporal_extrapolate()

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"scales={self.scales}, gamma={self.gamma_scale}"

    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Design block of the effect for the input ``x_col``; ``state_in`` carries the causal state of a previous chunk (see the class docstring)."""
        x3, shape, layout = to_btd(x_col, True)
        x_in = x3[..., 0]  # (B, T)
        B, T = x_in.shape
        dtype = x_in.dtype
        device = x_in.device

        S_len = len(self.scales)
        alphas = torch.tensor([2.0 / (float(s) + 1.0) for s in self.scales], dtype=torch.float64, device=device)  # (S,)
        gammas = 1.0 - alphas  # (S,)

        ema_prev = _carried(state_in, "ema", (B, S_len), device, dtype)
        env_prev = _carried(state_in, "env", (B, S_len), device, dtype)
        if ema_prev is None or env_prev is None:
            ema_prev = x_in[:, :1].expand(B, S_len).clone()
            env_prev = torch.zeros((B, S_len), dtype=dtype, device=device)

        # m_t = g m_{t-1} + a x_t, started from the carried value (the first input without a history)
        ema_seq = causal_linear_recurrence(x_in.unsqueeze(-1).expand(B, T, S_len), gammas, alphas, ema_prev)  # (B, T, S)
        band_seq = x_in.unsqueeze(-1) - ema_seq  # (B, T, S)
        band_sq = torch.tanh(0.25 * band_seq * band_seq)
        env_seq = causal_linear_recurrence(band_sq, gammas, alphas, env_prev)

        self._state_out = {
            "ema": ema_seq[:, -1, :].detach().clone(),
            "env": env_seq[:, -1, :].detach().clone(),
        }

        out = torch.cat([ema_seq, band_seq, env_seq], dim=-1)  # (B, T, 3*S)
        return from_btk(out, shape, layout)

    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        dtype = torch.get_default_dtype()
        weights = []
        for kind_mult in (1.0, 1.5, 2.0):
            for s in self.scales:
                weights.append(kind_mult * (float(s) ** self.gamma_scale))
        w_t = torch.tensor(weights, dtype=dtype, device=TORCH_DEVICE)
        return self.lambda_p * torch.diag(w_t)


class StructuralPriorEffect(BaseEffect):
    r"""
    Structural prior dictionary (internal to `foundation`; no formula token).

    A fixed dictionary of ``n_struct`` shape functions of the current value, drawn from four families that cover the usual
    structures of a response curve (the idea of a synthetic structural prior, Müller et al. 2022, here without any network):
    saturating sigmoids, softplus hinges, skewed Gaussian pulses and rational heavy-tail components. The pulses and the
    heavy-tail components are ordered by their centre, so a second-difference penalty applies along each of those two blocks.
    """

    stateful = False

    def __init__(
        self,
        feature_name: str,
        n_struct: int = 16,
        lambda_p: float = 1e-2,
        seed: int = 42,
        extrapolate: str = "continue",
    ):
        super().__init__(
            feature_name=feature_name,
            effect_type="struct_prior",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.n_struct = max(int(n_struct), 4)
        self.seed = int(seed)
        self.n_coeffs = self.n_struct

        # Partition n_struct into 4 canonical families:
        # 1) saturating monotone sigmoids, 2) softplus degree-day hinges,
        # 3) asymmetric Gaussian pulses, 4) rational Cauchy heavy-tail components.
        n1 = self.n_struct // 4
        n2 = self.n_struct // 4
        n3 = self.n_struct // 4
        n4 = self.n_struct - (n1 + n2 + n3)
        self._counts = (n1, n2, n3, n4)

        self._centers_1 = torch.linspace(-0.85, 0.85, max(n1, 1), dtype=torch.float64)
        self._slopes_1 = torch.linspace(2.0, 6.0, max(n1, 1), dtype=torch.float64)

        self._centers_2 = torch.linspace(-0.75, 0.75, max(n2, 1), dtype=torch.float64)
        self._signs_2 = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(max(n2, 1))], dtype=torch.float64)

        self._centers_3 = torch.linspace(-0.8, 0.8, max(n3, 1), dtype=torch.float64)      # ascending: the second difference runs along them
        self._widths_3 = torch.linspace(0.2, 0.5, max(n3, 1), dtype=torch.float64)

        self._centers_4 = torch.linspace(-0.7, 0.7, max(n4, 1), dtype=torch.float64)      # ascending as well
        self._scales_4 = torch.linspace(0.25, 0.6, max(n4, 1), dtype=torch.float64)
        self._state_out: Optional[Dict[str, Any]] = None

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"n_struct={self.n_struct}"

    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Design block of the effect for the input ``x_col``; ``state_in`` carries the causal state of a previous chunk (see the class docstring)."""
        x3, shape, layout = to_btd(x_col, True)
        x_in = x3[..., 0]
        dtype = x_in.dtype
        device = x_in.device
        x_exp = x_in.unsqueeze(-1)  # (B, T, 1)
        n1, n2, n3, n4 = self._counts
        cols = []

        if n1 > 0:
            c1 = self._centers_1[:n1].to(device=device, dtype=dtype).view(1, 1, n1)
            s1 = self._slopes_1[:n1].to(device=device, dtype=dtype).view(1, 1, n1)
            cols.append(torch.tanh(s1 * (x_exp - c1)))
        if n2 > 0:
            c2 = self._centers_2[:n2].to(device=device, dtype=dtype).view(1, 1, n2)
            sg2 = self._signs_2[:n2].to(device=device, dtype=dtype).view(1, 1, n2)
            cols.append(torch.nn.functional.softplus(4.0 * sg2 * (x_exp - c2)) / 4.0)
        if n3 > 0:
            c3 = self._centers_3[:n3].to(device=device, dtype=dtype).view(1, 1, n3)
            w3 = self._widths_3[:n3].to(device=device, dtype=dtype).view(1, 1, n3)
            z3 = (x_exp - c3) / w3
            # Skewed pulse: exp(-z^2 / 2) * sigmoid(1.5 * z)
            cols.append(torch.exp(-0.5 * z3 * z3) * torch.sigmoid(1.5 * z3))
        if n4 > 0:
            c4 = self._centers_4[:n4].to(device=device, dtype=dtype).view(1, 1, n4)
            sc4 = self._scales_4[:n4].to(device=device, dtype=dtype).view(1, 1, n4)
            z4 = (x_exp - c4) / sc4
            cols.append(z4 / (1.0 + z4 * z4))

        return from_btk(torch.cat(cols, dim=-1), shape, layout)

    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        dtype = torch.get_default_dtype()
        n1, n2, n3, n4 = self._counts
        blocks = [
            torch.eye(n1, dtype=dtype, device=TORCH_DEVICE) * 1.0,
            torch.eye(n2, dtype=dtype, device=TORCH_DEVICE) * 1.2,
            2.0 * _build_d2_penalty(n3, TORCH_DEVICE, dtype),
            1.5 * _build_d2_penalty(n4, TORCH_DEVICE, dtype),
        ]
        return self.lambda_p * torch.block_diag(*[b for b in blocks if b.shape[0] > 0])


# Name used by the benchmark scripts of the prototype.
PFNRidgeEffect = StructuralPriorEffect


class FoundationEffect(BaseEffect):
    r"""
    Universal Foundation Basis (`foundation()`).

    Combines four closed-form components of the same input into one basis with a block-diagonal penalty and exact state
    continuation: the causal patch basis (`PatchBasisEffect`: random-feature patch tokens with reversible instance statistics,
    Rahimi and Recht 2007; Nie et al. 2023; Kim et al. 2022, and Lévy-area hysteresis tokens, Lyons 1998), a quantile soft
    histogram (`QuantileHistEffect`), a multi-rate filterbank (`MultiRateFilterbankEffect`) and a structural prior dictionary
    (`StructuralPriorEffect`). No weight is trained: the coefficients are solved by P-WLS like those of any other effect.

    The four blocks have their own smoothing parameter (penalty coordinates ``patch``, ``hist``, ``mrfb``, ``struct``).
    """

    stateful = True
    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        patch_len: int = 16,
        n_tokens: int = 16,
        n_bins: int = 16,
        scales: Sequence[int] = (1, 4, 16, 64),
        n_struct: Optional[int] = None,
        include_levy: bool = True,
        lambda_p: float = 1e-2,
        seed: int = 42,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
        n_pfn: Optional[int] = None,
    ):
        if n_struct is not None and n_pfn is not None and int(n_struct) != int(n_pfn):
            raise ValueError(f"n_struct={n_struct} and n_pfn={n_pfn} disagree: n_pfn is an alias of n_struct, give one of them")
        n_struct = int(n_struct if n_struct is not None else (n_pfn if n_pfn is not None else 16))
        super().__init__(
            feature_name=feature_name,
            effect_type="foundation",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        if additional_features:
            self.input_features = [feature_name] + [f for f in additional_features if f != feature_name]
        else:
            self.input_features = [feature_name]

        self.patch_effect = PatchBasisEffect(
            feature_name=feature_name,
            patch_len=patch_len,
            n_tokens=n_tokens,
            include_levy=include_levy,
            lambda_p=lambda_p,
            seed=seed,
            extrapolate=extrapolate,
            additional_features=additional_features,
        )
        self.hist_effect = QuantileHistEffect(
            feature_name=feature_name,
            n_bins=n_bins,
            window=max(patch_len, 8),
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.mrfb_effect = MultiRateFilterbankEffect(
            feature_name=feature_name,
            scales=scales,
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.struct_effect = StructuralPriorEffect(
            feature_name=feature_name,
            n_struct=n_struct,
            lambda_p=lambda_p,
            seed=seed,
            extrapolate=extrapolate,
        )
        self.n_struct = self.struct_effect.n_struct

        self.sub_effects: List[BaseEffect] = [
            self.patch_effect,
            self.hist_effect,
            self.mrfb_effect,
            self.struct_effect,
        ]
        self.n_coeffs = sum(e.get_n_coeffs() for e in self.sub_effects)
        self._state_out: Optional[Dict[str, Any]] = None
        self._check_temporal_extrapolate()

    @property
    def lambda_p(self) -> float:
        """The smoothing weight of the effect: the one of the first block (all four are equal unless set apart)."""
        subs = getattr(self, "sub_effects", None)
        return subs[0].lambda_p if subs else self._lambda_p

    @lambda_p.setter
    def lambda_p(self, value: float) -> None:
        """The smoothing weight(s) of the effect."""
        self._lambda_p = float(value)
        for e in getattr(self, "sub_effects", []):
            e.lambda_p = self._lambda_p

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def coordinate_labels(self) -> List[str]:
        """Names of the penalty coordinates, in the order of ``penalty_coordinates()``."""
        return ["patch", "hist", "mrfb", "struct"]

    def penalty_coordinates(self) -> list:
        """One smoothing weight (lambda, not log10) per block: patch, hist, mrfb, struct."""
        return [e.lambda_p for e in self.sub_effects]

    def set_penalty_coordinates(self, values: Sequence[float]) -> None:
        """Sets the weights of the four blocks; a list of length 1 sets all four."""
        values = [float(v) for v in values]
        n = len(self.sub_effects)
        if len(values) == 1:
            values = values * n
        if len(values) != n:
            raise ValueError(f"{self.__class__.__name__} has {n} penalty coordinates, got {len(values)}")
        for e, v in zip(self.sub_effects, values):
            e.lambda_p = v

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return (
            f"patch={self.patch_effect.get_n_coeffs()}, "
            f"hist={self.hist_effect.get_n_coeffs()}, "
            f"mrfb={self.mrfb_effect.get_n_coeffs()}, "
            f"struct={self.struct_effect.get_n_coeffs()}"
        )

#: <foundation_feature_map>
    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Design block of the effect for the input ``x_col``; ``state_in`` carries the causal state of a previous chunk (see the class docstring)."""
        x3, shape, layout = to_btd(x_col, True)
        x_primary = x3[..., :1]

        def sub(prefix_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
            """The part of the carried state that belongs to one sub-effect."""
            if state_in is None:
                return None
            picked = {inner: state_in[outer] for outer, inner in prefix_map.items() if outer in state_in}
            return picked or None

        phi_patch = self.patch_effect.build_feature_map(x3, state_in=sub({"patch": "hist"}))
        phi_hist = self.hist_effect.build_feature_map(x_primary, state_in=sub({"qhist": "hist"}))
        phi_mrfb = self.mrfb_effect.build_feature_map(x_primary, state_in=sub({"ema": "ema", "env": "env"}))
        phi_struct = self.struct_effect.build_feature_map(x_primary)
        self._state_out = {
            "patch": self.patch_effect._state_out["hist"],
            "qhist": self.hist_effect._state_out["hist"],
            "ema": self.mrfb_effect._state_out["ema"],
            "env": self.mrfb_effect._state_out["env"],
        }
        return from_btk(torch.cat([phi_patch, phi_hist, phi_mrfb, phi_struct], dim=-1), shape, layout)
#: </foundation_feature_map>

#: <foundation_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        blocks = [e.build_penalty_matrix() for e in self.sub_effects]
        return torch.block_diag(*blocks)
#: </foundation_penalty_matrix>
