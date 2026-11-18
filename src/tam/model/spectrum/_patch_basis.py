# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Causal RevIN patch basis (`patch_basis`): causal patch tokens with running instance statistics and optional
Level-2 antisymmetric Lévy area hysteresis tokens.
"""

from typing import Optional, List, Dict, Any
import math
import torch
from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, from_btk, tail_state, to_btd


class PatchBasisEffect(BaseEffect):
    r"""
    Concept A: Causal Patch-Tokenized Basis with RevIN Scale Tokens and Level-2
    Antisymmetric Lévy Area Hysteresis Tokens (`patch_basis`).
    """

    stateful = True

    def __init__(
        self,
        feature_name: str,
        patch_len: int = 16,
        n_tokens: int = 16,
        include_levy: bool = True,
        lambda_p: float = 1e-2,
        seed: int = 42,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
    ):
        super().__init__(
            feature_name=feature_name,
            effect_type="patch_basis",
            lambda_p=lambda_p,
            extrapolate=extrapolate,
        )
        self.patch_len = max(int(patch_len), 2)
        self.n_tokens = max(int(n_tokens), 2)
        self.include_levy = bool(include_levy)
        self.seed = int(seed)

        if additional_features:
            self.input_features = [feature_name] + [f for f in additional_features if f != feature_name]
        else:
            self.input_features = [feature_name]

        self.d_in = len(self.input_features)
        # Number of antisymmetric Levy area tokens:
        # With several input columns there is one Levy area per pair of columns; with one column and include_levy there is one
        # self-lag area (the series against its own previous value), which captures rising versus falling hysteresis loops.
        if self.include_levy:
            self.n_levy = max(self.d_in * (self.d_in - 1) // 2, 1)
        else:
            self.n_levy = 0

        # Total columns: n_tokens random Fourier patch features + 2 RevIN stats (mean, log-std) + n_levy
        self.n_coeffs = self.n_tokens + 2 + self.n_levy

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        raw_w = torch.randn((self.n_tokens, self.patch_len), generator=gen, dtype=torch.float64)
        q_mat, _ = torch.linalg.qr(raw_w.mT, mode="reduced")
        q_mat = q_mat.mT  # (min(M, P), P)
        if q_mat.shape[0] < self.n_tokens:
            extra = torch.randn((self.n_tokens - q_mat.shape[0], self.patch_len), generator=gen, dtype=torch.float64)
            extra = extra / torch.clamp(torch.linalg.norm(extra, dim=1, keepdim=True), min=1e-8)
            q_mat = torch.cat([q_mat, extra], dim=0)
        self._w_patch = q_mat[:self.n_tokens, :] * (1.5 / math.sqrt(float(self.patch_len)))
        self._b_patch = torch.rand((self.n_tokens,), generator=gen, dtype=torch.float64) * (2.0 * math.pi)
        self._state_out: Optional[Dict[str, Any]] = None
        self._check_temporal_extrapolate()

    def get_n_coeffs(self) -> int:
        """Number of columns of the design block of this effect."""
        return self.n_coeffs

    def summary_details(self) -> str:
        """Short description of the settings of the effect, for the model summary."""
        return f"patch_len={self.patch_len}, n_tokens={self.n_tokens}, levy={self.n_levy}"

#: <patch_basis_feature_map>
    def build_feature_map(
        self,
        x_col: torch.Tensor,
        state_in: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Design block of the effect for the input ``x_col``; ``state_in`` carries the causal state of a previous chunk (see the class docstring)."""
        x_in, shape, layout = to_btd(x_col, self.multi_input)
        B, T, D = x_in.shape
        P = self.patch_len
        dtype = x_in.dtype
        device = x_in.device

        padded = causal_history(x_in, P - 1, state_in)  # (B, T + P - 1, D)
        self._state_out = {"hist": tail_state(padded, P - 1)}

        # Primary channel patches: shape (B, T, P)
        primary_padded = padded[:, :, 0]
        patches = primary_padded.unfold(dimension=1, size=P, step=1)  # (B, T, P)

        # Causal RevIN local statistics
        mu_p = torch.mean(patches, dim=-1, keepdim=True)  # (B, T, 1)
        var_p = torch.mean((patches - mu_p) ** 2, dim=-1, keepdim=True)
        sigma_p = torch.sqrt(var_p + 1e-6)  # (B, T, 1)
        patches_norm = (patches - mu_p) / sigma_p  # (B, T, P)

        w_patch = self._w_patch.to(device=device, dtype=dtype)  # (M, P)
        b_patch = self._b_patch.to(device=device, dtype=dtype)  # (M,)
        proj = torch.einsum("btp,mp->btm", patches_norm, w_patch) + b_patch.view(1, 1, -1)
        phi_tokens = math.sqrt(2.0 / float(self.n_tokens)) * torch.cos(proj)  # (B, T, M)

        revin_tokens = torch.cat([
            torch.tanh(0.25 * mu_p),
            torch.tanh(0.5 * torch.log(sigma_p)),
        ], dim=-1)  # (B, T, 2)

        parts = [phi_tokens, revin_tokens]

        if self.n_levy > 0:
            # Windows across all D channels: (B, T, D, P)
            all_patches = padded.unfold(dimension=1, size=P, step=1)
            # Local center each channel over the patch
            all_centered = all_patches - torch.mean(all_patches, dim=-1, keepdim=True)
            all_std = torch.sqrt(torch.mean(all_centered ** 2, dim=-1, keepdim=True) + 1e-6)
            u_path = all_centered / all_std  # (B, T, D, P)
            du = u_path[..., 1:] - u_path[..., :-1]  # (B, T, D, P-1)
            u_mid = 0.5 * (u_path[..., 1:] + u_path[..., :-1]) - u_path[..., :1]  # (B, T, D, P-1)

            levy_list = []
            if D >= 2:
                for a in range(D):
                    for b in range(a + 1, D):
                        area_ab = 0.5 * torch.mean(
                            u_mid[:, :, a, :] * du[:, :, b, :] - u_mid[:, :, b, :] * du[:, :, a, :],
                            dim=-1,
                            keepdim=True,
                        )
                        levy_list.append(torch.tanh(area_ab))
            else:
                # Hoff Lead-Lag self-hysteresis area between x_s and its 1-step delayed path
                u_lead = u_path[:, :, 0, 1:]
                u_lag = u_path[:, :, 0, :-1]
                du_lead = u_lead[..., 1:] - u_lead[..., :-1]
                du_lag = u_lag[..., 1:] - u_lag[..., :-1]
                mid_lead = 0.5 * (u_lead[..., 1:] + u_lead[..., :-1]) - u_lead[..., :1]
                mid_lag = 0.5 * (u_lag[..., 1:] + u_lag[..., :-1]) - u_lag[..., :1]
                area_self = 0.5 * torch.mean(mid_lead * du_lag - mid_lag * du_lead, dim=-1, keepdim=True)
                levy_list.append(torch.tanh(area_self))

            parts.append(torch.cat(levy_list, dim=-1))

        return from_btk(torch.cat(parts, dim=-1), shape, layout)
#: </patch_basis_feature_map>

    def _token_roughness_weights(self) -> torch.Tensor:
        r"""Penalty weight of each patch-token column: ``1 + r_m / mean(r)`` with ``r_m`` the roughness of its filter.

        The roughness is the squared norm of the second difference of the filter taken along the patch axis. A smooth filter
        keeps a weight near 1, an oscillating one a larger weight; the mean keeps the overall scale close to a plain ridge.
        With a patch shorter than 3 steps there is no second difference and every weight is 1.
        """
        w = torch.ones(self.n_tokens, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
        if self.patch_len < 3:
            return w
        rough = (self._w_patch[:, 2:] - 2.0 * self._w_patch[:, 1:-1] + self._w_patch[:, :-2]).pow(2).sum(dim=1)
        mean = float(rough.mean())
        if mean <= 1e-12:
            return w
        return 1.0 + (rough / mean).to(device=TORCH_DEVICE, dtype=w.dtype)

#: <patch_basis_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Penalty matrix of the effect, scaled by its smoothing weights (see the class docstring)."""
        weights = torch.ones(self.n_coeffs, dtype=torch.get_default_dtype(), device=TORCH_DEVICE)
        weights[: self.n_tokens] = self._token_roughness_weights()   # the RevIN statistics and the Levy-area tokens keep a ridge
        return self.lambda_p * torch.diag(weights)
#: </patch_basis_penalty_matrix>
