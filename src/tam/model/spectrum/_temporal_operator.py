# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Form B Stateful Temporal Operators on Evaluated Basis Matrices.

Instead of treating sequence-mixing mechanisms (Attention, HiPPO, SSA) as
standalone column-wise feature effects on raw scalar covariates, this module
implements causal temporal operators that act directly on evaluated inner basis
feature maps:
    \Phi_{\text{mixed}}(X) = \mathcal{T}_{\text{op}}\big(\Phi_{\text{base}}(X)\big) \in \mathbb{R}^{T \times K_{\text{out}}}
with exact causal state continuation (`state_in` / `_state_out`) across arbitrary
chunk boundaries:
    \|\Phi_{[0, m)}[n:m] - \Phi_{[n, m)}(\text{state\_in}=\mathcal{S}_n)\|_\infty \le 10^{-10}.
"""

from typing import Optional, List, Any, Dict
import math
import torch
from ._base_effects import BaseEffect
from ._penalty_coordinates import CoordinatePenaltyMixin
from ._temporal_utils import causal_history, causal_linear_recurrence, causal_window_unfold, from_btk, tail_state, to_btd
from ._spline import SplineEffect
from ._fourier import FourierEffect
from ._linear import LinearEffect


class TemporalOperatorEffect(CoordinatePenaltyMixin, BaseEffect):
    r"""
    Form B Stateful Temporal Operator wrapping an inner functional basis.

    Evaluates an inner basis \Phi_{\text{base}}(X) \in \mathbb{R}^{B \times T \times K_{\text{base}}}
    and applies a causal temporal mixing operator along the time axis:
      - `operator_type='attn'`: Causal exponential-decay FAVOR+ positive orthogonal random
        feature attention mixing across past basis evaluations.
      - `operator_type='hippo'`: Causal multi-rate continuous-time state-space memory filter
        mixing each basis column with its historical trajectory.
      - `operator_type='ssa'`: Causal sliding-window Hankel projection of the evaluated basis.

    Concatenates the immediate evaluated basis \Phi_{\text{base}}(X)_t with its causally
    mixed temporal context \mathcal{T}_{\text{op}}(\Phi_{\text{base}})_t so the Primal solver
    simultaneously estimates instantaneous and memory-modulated coefficients.
    """

    stateful = True
    keeps_feature_dim = True

    def __init__(
        self,
        feature_name: str,
        operator_type: str = "attn",
        inner_basis: str = "spline",
        k_base: int = 8,
        decay: float = 0.92,
        n_random_features: int = 8,
        window: int = 12,
        lambda_p: float = 1e-2,
        seed: int = 42,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
    ):
        super().__init__(
            feature_name=feature_name,
            effect_type="temporal_operator",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.operator_type = str(operator_type).lower().strip()
        if self.operator_type not in ("attn", "hippo", "ssa"):
            raise ValueError(f"Unsupported operator_type='{operator_type}'. Choose from 'attn', 'hippo', 'ssa'.")

        self.inner_basis_type = str(inner_basis).lower().strip()
        self.k_base = max(int(k_base), 2)
        self.decay = float(min(max(decay, 0.01), 0.999))
        self.n_random_features = max(int(n_random_features), 2)
        self.window = max(int(window), 2)
        self.seed = int(seed)

        if additional_features:
            self.input_features = [feature_name] + [f for f in additional_features if f != feature_name]
        else:
            self.input_features = [feature_name]

        if self.inner_basis_type in ("s", "spline"):
            self.inner_effect: BaseEffect = SplineEffect(
                feature_name,
                max(self.k_base, 4),
                3,
                2,
                lambda_p,
                "linear",
                1,
            )
        elif self.inner_basis_type in ("f", "fourier"):
            self.inner_effect = FourierEffect(
                feature_name,
                max(self.k_base // 2, 1),
                1,
                lambda_p,
                False,
                "continue",
            )
        else:
            self.inner_effect = LinearEffect(
                feature_name,
                math.pi,
                lambda_p,
                "continue",
            )

        self.k_inner = self.inner_effect.get_n_coeffs()
        # Total coefficients: k_inner instantaneous + k_inner temporally mixed
        self.n_coeffs = 2 * self.k_inner
        # the base block and the operator block have their own smoothing weight
        self._init_coordinates(["base", "op"], [1.0, 1.0])

        # Deterministic orthogonal projection for 'attn'
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        raw_w = torch.randn((self.k_inner, self.n_random_features), generator=gen, dtype=torch.float64)
        q_mat, _ = torch.linalg.qr(raw_w, mode="reduced")
        if q_mat.shape[1] < self.n_random_features:
            pad = torch.randn((self.k_inner, self.n_random_features - q_mat.shape[1]), generator=gen, dtype=torch.float64)
            q_mat = torch.cat([q_mat, pad], dim=1)
        self._proj_w = q_mat[:, :self.n_random_features] * 0.5  # (K_inner, M)

        # Multi-rate state weights for 'hippo'
        rates = torch.linspace(0.05, 0.5, self.k_inner, dtype=torch.float64)
        self._hippo_gammas = 1.0 - rates  # (K_inner,)

        # Smooth temporal filter weights for 'ssa'
        lags = torch.arange(self.window, dtype=torch.float64)
        ssa_w = torch.exp(-lags / max(float(self.window) * 0.35, 1.0))
        self._ssa_weights = ssa_w / torch.sum(ssa_w)  # (W,)

        self._state_out: Optional[Dict[str, Any]] = None
        self._check_temporal_extrapolate()

    def initialize(self, x_cols: torch.Tensor) -> None:
        """Initializes the inner basis on the primary input (every group); the operators hold no data statistic of their own."""
        x3, _, _ = to_btd(x_cols, True)
        if x3.numel() > 0:
            self.inner_effect.initialize(x3[..., 0])

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"op={self.operator_type}, inner={self.inner_basis_type}(k={self.k_inner}), decay={self.decay:.2f}"

    def _eval_inner_basis(self, x3: torch.Tensor) -> torch.Tensor:
        """The inner basis on the primary input, ``(B, T, K)``. The inner effects are pointwise: no state."""
        x_primary = x3[..., 0] if not self.inner_effect.multi_input else x3
        phi = self.inner_effect.transform(x_primary).to(dtype=x3.dtype)   # some inner bases build in the default dtype
        return phi if phi.dim() == 3 else phi.unsqueeze(0)

#: <temp_op_feature_map>
    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """``(B, T, d)`` in, ``(B, T, 2 K)`` out: the inner basis and its causal temporal mix, one group at a time.

        Every operator is a linear recurrence along time (``attn``, ``hippo``) or a window over a carried history (``ssa``):
        time and memory are linear in ``T``. The state is per group: ``{"S", "Z"}`` (attn), ``{"h"}`` (hippo), ``{"hist"}`` (ssa).
        """
        x3, shape, layout = to_btd(x_col, True)
        phi_base = self._eval_inner_basis(x3)  # (B, T, K_inner)
        B, T, K = phi_base.shape
        dtype = phi_base.dtype
        device = phi_base.device

        def carried(key: str, expected: tuple) -> Optional[torch.Tensor]:
            """The carried state ``key`` when it is there and has the expected shape, else None."""
            if state_in is None or key not in state_in:
                return None
            s = state_in[key].to(device=device, dtype=dtype)
            if tuple(s.shape) != expected:
                raise ValueError(f"state '{key}' has shape {tuple(s.shape)}, expected {expected}: continue_state needs the same groups and features as training")
            return s

        if self.operator_type == "attn":
            proj_w = self._proj_w.to(device=device, dtype=dtype)  # (K, M)
            # Bounded positive random features phi_rf(u) = exp(u @ W - ||u||^2 / 2)
            u_norm = torch.clamp(phi_base, -4.0, 4.0)
            proj = u_norm @ proj_w  # (B, T, M)
            norm_sq = 0.5 * torch.sum(u_norm * u_norm, dim=-1, keepdim=True) / max(float(K), 1.0)
            phi_rf = torch.exp(torch.clamp(proj - norm_sq, min=-10.0, max=6.0))  # (B, T, M)

            # Causal discounted prefix accumulators (the random-feature products and their normalizer), solved as linear recurrences along time
            M = self.n_random_features
            S_prev = carried("S", (B, M, K))
            Z_prev = carried("Z", (B, M))
            kv_outer = torch.einsum("btm,btk->btmk", phi_rf, phi_base)  # (B, T, M, K)
            S_seq = causal_linear_recurrence(
                kv_outer.reshape(B, T, M * K), self.decay, 1.0 - self.decay, None if S_prev is None else S_prev.reshape(B, M * K)
            ).reshape(B, T, M, K)
            Z_seq = causal_linear_recurrence(phi_rf, self.decay, 1.0 - self.decay, Z_prev)

            num = torch.einsum("btm,btmk->btk", phi_rf, S_seq)
            den = torch.clamp(torch.einsum("btm,btm->bt", phi_rf, Z_seq).unsqueeze(-1), min=1e-8)
            phi_mixed = num / den
            self._state_out = {"S": S_seq[:, -1].detach().clone(), "Z": Z_seq[:, -1].detach().clone()}

        elif self.operator_type == "hippo":
            gammas = self._hippo_gammas.to(device=device, dtype=dtype)  # (K,)
            h_prev = carried("h", (B, K))
            if h_prev is None:
                h_prev = phi_base[:, 0, :].clone()
            phi_mixed = causal_linear_recurrence(phi_base, gammas, 1.0 - gammas, h_prev)
            self._state_out = {"h": phi_mixed[:, -1].detach().clone()}

        else:  # operator_type == "ssa"
            W = self.window
            w_ssa = self._ssa_weights.to(device=device, dtype=dtype)  # (W,)
            x_hist = causal_history(phi_base, W - 1, state_in, key="hist")
            windows = causal_window_unfold(x_hist, W, t_len=T)  # (B, T, W, K), oldest step first
            phi_mixed = torch.einsum("btwk,w->btk", torch.flip(windows, dims=[2]), w_ssa)   # lag 0 weighs w_ssa[0]
            self._state_out = {"hist": tail_state(x_hist, W - 1)}

        out = torch.cat([phi_base, phi_mixed], dim=-1)  # (B, T, 2*K)
        return from_btk(out, shape, layout)
#: </temp_op_feature_map>

    def _inner_penalty(self, lam: float) -> torch.Tensor:
        """The penalty of the inner basis at the smoothing weight ``lam``."""
        saved = self.inner_effect.lambda_p
        self.inner_effect.lambda_p = lam
        try:
            P_inner = self.inner_effect.build_penalty_matrix()
        finally:
            self.inner_effect.lambda_p = saved
        return P_inner.to_dense() if P_inner.is_sparse else P_inner

#: <temp_op_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        lam_base, lam_op = self.penalty_coordinates()
        P_base = self._inner_penalty(lam_base)
        P_op = self._inner_penalty(lam_op)
        K = self.k_inner
        dtype = P_base.dtype
        device = P_base.device
        P = torch.zeros((2 * K, 2 * K), dtype=dtype, device=device)
        P[:K, :K] = P_base
        # Slightly stronger regularization on temporal operator channels
        P[K:, K:] = P_op + lam_op * torch.eye(K, dtype=dtype, device=device)
        return P
#: </temp_op_penalty_matrix>
