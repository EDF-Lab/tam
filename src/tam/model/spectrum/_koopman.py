# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Koopman Operator Effect via Extended / Kernel Dynamic Mode Decomposition (EDMD / kDMD).

Linearizes finite-dimensional non-linear dynamical systems :math:`x_{t+1} = F(x_t)`
by learning empirical Koopman eigenfunctions :math:`\varphi_k(x)` satisfying:

.. math::

    (\mathcal{K} \varphi_k)(x_t) = \varphi_k(x_{t+1}) = \mu_k \varphi_k(x_t)

using dictionary lifting (Hankel delay embedding, quadratic monomials, and Random Fourier Features)
projected through SVD-EDMD with residual-consistent penalty regularization.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, from_btk, tail_state, to_btd


class KoopmanEffect(BaseEffect):
    r"""Global linearization of non-linear dynamical systems via Koopman Operator Decomposition.

    The effect has exactly ``k`` columns. The modes are kept in the order of their transition residual; a complex-conjugate
    pair is kept whole (its real and imaginary parts are two neighbouring columns), so when only one column is left the pair is
    skipped and the next real mode is taken. When the data hold fewer than ``k`` modes (numerical rank), the remaining columns
    are orthonormal directions of the dictionary, with a very large penalty.

    Parameters:
        feature_name: Primary covariate column name, or comma-separated column names.
        k: Number of empirical Koopman modes to retain (the number of columns).
        n_lift: Minimum size of the lifted observable dictionary.
        delay_embed: Number of causal time-delay Hankel embedding steps (default 1).
        reg: Tikhonov regularization for SVD truncation.
        alpha_res: Penalty weight on Koopman transition residuals.
        alpha_unit: Penalty weight on divergence from unit circle.
        seed: Random seed for reproducible Fourier features.
        lambda_p: Smoothing parameter.
        extrapolate: Out-of-distribution extrapolation mode.
        additional_features: Further covariate columns read with the primary one.
    """

    keeps_feature_dim: bool = True
    input_features: Optional[List[str]] = None

    #: Penalty of a column that is not a Koopman mode (a filler direction): the solver leaves it at zero.
    FILLER_PENALTY = 1e4

    def __init__(
        self,
        feature_name: Union[str, Sequence[str]],
        k: int = 12,
        n_lift: int = 64,
        delay_embed: int = 1,
        reg: float = 1e-4,
        alpha_res: float = 10.0,
        alpha_unit: float = 10.0,
        seed: int = 42,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
    ):
        if isinstance(feature_name, (list, tuple)):
            features = [str(f).strip() for f in feature_name]
            primary_name = features[0]
        elif "," in str(feature_name):
            features = [s.strip() for s in str(feature_name).split(",") if s.strip()]
            primary_name = features[0]
        else:
            primary_name = str(feature_name).strip()
            features = [primary_name]
        features = features + [f for f in (additional_features or []) if f not in features]

        super().__init__(
            feature_name=primary_name,
            effect_type="koopman",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.input_features = features
        self.k = max(1, int(k))
        self.n_lift = int(n_lift)
        self.delay_embed = max(1, int(delay_embed))
        self.reg = float(reg)
        self.alpha_res = float(alpha_res)
        self.alpha_unit = float(alpha_unit)
        self.seed = int(seed)

        self._W_proj: Optional[torch.Tensor] = None
        self._penalty_diag: Optional[torch.Tensor] = None
        self._mode_sizes: List[int] = []
        self._omega: Optional[torch.Tensor] = None
        self._triu_i: Optional[torch.Tensor] = None
        self._triu_j: Optional[torch.Tensor] = None
        self._d_in: int = len(features)
        self._d_hankel: int = self._d_in * self.delay_embed
        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self._n_hist = self.delay_embed - 1
        if self.stateful:
            self._check_temporal_extrapolate()

    @property
    def stateful(self) -> bool:
        """With a delay embedding the effect reads the last ``delay_embed - 1`` rows of the previous pass."""
        return self.delay_embed > 1

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"k={self.k}, d_H={self.delay_embed}, lift={self.n_lift}"

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.k

    def _hankel(self, x3: torch.Tensor, state_in: Optional[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal time-delay embedding of ``(B, T, d)``: ``(B, T, d * delay_embed)`` and the history to carry."""
        B, T, d_in = x3.shape
        x_hist = causal_history(x3, self._n_hist, state_in)
        d_H = self.delay_embed
        blocks = [x_hist[:, d_H - 1 - lag: d_H - 1 - lag + T, :] for lag in range(d_H)]
        return torch.cat(blocks, dim=-1), tail_state(x_hist, self._n_hist)

    def _setup_lifting_dictionary(self, d_h: int) -> None:
        """Quadratic monomial indices and the Random Fourier Feature projection (no constant column), drawn on the CPU in float64."""
        self._triu_i, self._triu_j = torch.triu_indices(d_h, d_h)
        n_quad = len(self._triu_i)
        n_base = d_h + n_quad

        n_rff = max(0, self.n_lift - n_base)
        if n_rff > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            self._omega = torch.randn(d_h, n_rff, generator=gen, dtype=torch.float64) * 0.5
        else:
            self._omega = None

    def _lift(self, x_h: torch.Tensor) -> torch.Tensor:
        """Evaluates non-linear observable lifting dictionary psi(x_h) without constant offset."""
        D = x_h.shape[-1]
        if self._triu_i is None or len(self._triu_i) != D * (D + 1) // 2:
            self._setup_lifting_dictionary(D)

        if self._x_mean is not None and self._x_std is not None and self._x_mean.shape[-1] == D:
            x_norm = (x_h - self._x_mean.to(x_h.device, x_h.dtype)) / (
                self._x_std.to(x_h.device, x_h.dtype) + 1e-6
            )
        else:
            x_norm = x_h

        parts = [x_norm]

        # Degree-2 monomials
        i_idx = self._triu_i.to(x_h.device)
        j_idx = self._triu_j.to(x_h.device)
        quad = x_norm[..., i_idx] * x_norm[..., j_idx]
        parts.append(quad)

        # Random Fourier Features (centered to avoid constant mode)
        if self._omega is not None:
            omega = self._omega.to(device=x_h.device, dtype=x_h.dtype)
            rff = torch.sin(x_norm @ omega)
            parts.append(rff)

        return torch.cat(parts, dim=-1)

    def _fallback(self, M: int, device, dtype) -> None:
        """Too little data or a failed decomposition: the first ``k`` dictionary directions, plainly penalized."""
        self._W_proj = torch.eye(M, self.k, device=device, dtype=dtype)
        self._penalty_diag = torch.ones(self.k, device=device, dtype=dtype)
        self._mode_sizes = [1] * self.k

    def initialize(self, x_cols: torch.Tensor) -> None:
        """Fits SVD-EDMD Koopman operator on the training tensor (every group, transitions inside a group) and keeps ``k`` columns."""
        x3, _, _ = to_btd(x_cols, True)
        self._d_in = x3.shape[-1]
        self._d_hankel = self._d_in * self.delay_embed
        x_h, _ = self._hankel(x3, None)

        x_h_flat = x_h.reshape(-1, x_h.shape[-1])
        if x_h_flat.shape[0] > 1:
            self._x_mean = x_h_flat.mean(dim=0, keepdim=True).detach()
            std_val = x_h_flat.std(dim=0, keepdim=True, unbiased=False).detach()
            self._x_std = torch.where(std_val > 1e-6, std_val, torch.ones_like(std_val))
        else:
            self._x_mean = None
            self._x_std = None

        self._triu_i = None
        device, dtype = x_cols.device, x_cols.dtype

        B, T, d_h = x_h.shape
        if T < 3:
            psi = self._lift(x_h)
            self._fallback(psi.shape[-1], device, dtype)
            return
        x_h_X = x_h[:, :-1, :].reshape(-1, d_h)
        x_h_Y = x_h[:, 1:, :].reshape(-1, d_h)

        # Subsample snapshots before lifting when dataset is large (> 20000) to avoid redundant dictionary evaluation
        if x_h_X.shape[0] > 20000:
            step = x_h_X.shape[0] // 20000
            x_h_X = x_h_X[::step]
            x_h_Y = x_h_Y[::step]

        psi_X = self._lift(x_h_X)
        psi_Y = self._lift(x_h_Y)
        M = psi_X.shape[-1]
        if self.k > M:
            raise ValueError(f"koopman: k={self.k} is larger than the lifted dictionary ({M} observables); raise n_lift or lower k")

        try:
            U, S, Vh = torch.linalg.svd(psi_X, full_matrices=False)
            V = Vh.mH

            s_max = S[0]
            cutoff = max(float(s_max) * 1e-5, self.reg)
            rank = int((S > cutoff).sum().item())
            r = min(max(rank, 1), self.k, M)

            U_r = U[:, :r]
            S_r = S[:r]
            V_r = V[:, :r]

            inv_S = (1.0 / (S_r + self.reg)).unsqueeze(-1)
            K_tilde = inv_S * (U_r.mH @ psi_Y @ V_r)

            evals, evecs = torch.linalg.eig(K_tilde)

            scaled_evecs = inv_S * evecs
            W_complex = V_r.to(scaled_evecs.dtype) @ scaled_evecs

            # Batched real BLAS projection replaces per-mode psi_X_c @ w calls
            W_re = torch.real(W_complex).to(psi_X.dtype)
            W_im = torch.imag(W_complex).to(psi_X.dtype)
            W_cat = torch.cat([W_re, W_im], dim=1)
            PX_re, PX_im = torch.split(psi_X @ W_cat, r, dim=1)
            PY_re, PY_im = torch.split(psi_Y @ W_cat, r, dim=1)

            evals_list = evals.detach().cpu().tolist()
            units = []   # (residual, |mu|, [columns]): a real mode has one column, a conjugate pair its real and imaginary parts
            visited_conjugates = set()

            for j in range(r):
                if j in visited_conjugates:
                    continue
                mu = evals_list[j]
                pw_x = torch.complex(PX_re[:, j], PX_im[:, j])
                pw_y = torch.complex(PY_re[:, j], PY_im[:, j])
                res_vec = pw_y - evals[j] * pw_x
                denom = float(torch.real(torch.vdot(pw_x, pw_x)).item()) + 1e-8
                res_sq = float(torch.real(torch.vdot(res_vec, res_vec)).item() / denom)
                mu_mag = float(abs(mu))

                if abs(float(mu.imag)) > 1e-4:
                    conj_idx = None
                    mu_conj = mu.conjugate()
                    for m in range(j + 1, r):
                        if m not in visited_conjugates and abs(evals_list[m] - mu_conj) < 1e-4:
                            conj_idx = m
                            break

                    norm_re = torch.norm(PX_re[:, j]) + 1e-8
                    norm_im = torch.norm(PX_im[:, j]) + 1e-8
                    units.append((res_sq, mu_mag, [(W_re[:, j] / norm_re).to(dtype), (W_im[:, j] / norm_im).to(dtype)]))

                    if conj_idx is not None:
                        visited_conjugates.add(conj_idx)
                else:
                    norm_re = torch.norm(PX_re[:, j]) + 1e-8
                    units.append((res_sq, mu_mag, [(W_re[:, j] / norm_re).to(dtype)]))

            units.sort(key=lambda item: item[0])

            # Fill k columns in the order of the residual; a pair that does not fit in the room left is skipped (never split)
            cols, pens, sizes = [], [], []
            for res_sq, mu_mag, vecs in units:
                if len(cols) + len(vecs) > self.k:
                    continue
                pen = 1.0 + self.alpha_res * res_sq + self.alpha_unit * ((mu_mag ** 2 - 1.0) ** 2)
                cols.extend(vecs)
                pens.extend([pen] * len(vecs))
                sizes.append(len(vecs))
                if len(cols) == self.k:
                    break

            if len(cols) < self.k:
                # Fewer modes than k (numerical rank): complete with orthonormal directions of the dictionary, penalized heavily
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self.seed + 1)
                span = torch.stack(cols, dim=-1).to(torch.float64).cpu() if cols else torch.zeros(M, 0, dtype=torch.float64)
                need = self.k - len(cols)
                cand = torch.randn(M, need + 4, generator=gen, dtype=torch.float64)
                if span.shape[1] > 0:
                    q_span, _ = torch.linalg.qr(span, mode="reduced")
                    cand = cand - q_span @ (q_span.mT @ cand)
                q_new, _ = torch.linalg.qr(cand, mode="reduced")
                for j in range(need):
                    cols.append(q_new[:, j].to(device=device, dtype=dtype))
                    pens.append(self.FILLER_PENALTY)
                    sizes.append(1)

            self._W_proj = torch.stack(cols, dim=-1)
            self._penalty_diag = torch.tensor(pens, device=device, dtype=dtype)
            self._mode_sizes = sizes

        except Exception:
            self._fallback(M, device, dtype)

#: <koopman_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Evaluates Koopman eigenfunction feature map along each trajectory: ``(B, T, d)`` in, ``(B, T, k)`` out."""
        if self._W_proj is None:
            self.initialize(x_col)

        x3, shape, layout = to_btd(x_col, True)
        if x3.shape[-1] != self._d_in:
            raise ValueError(f"koopman expects {self._d_in} input column(s), got {x3.shape[-1]}")
        x_h, hist = self._hankel(x3, state_in)
        if self.stateful:
            self._state_out = {"hist": hist}
        psi = self._lift(x_h)

        W = self._W_proj.to(device=psi.device, dtype=psi.dtype)
        return from_btk(psi @ W, shape, layout)
#: </koopman_feature_map>

#: <koopman_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Returns diagonal penalty matrix weighted by Koopman spectral residuals and lambda_p."""
        if self._penalty_diag is None:
            self._penalty_diag = torch.ones(self.k, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

        return torch.diag(self.lambda_p * self._penalty_diag.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype()))
#: </koopman_penalty_matrix>
