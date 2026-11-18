# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Linearized self-attention Primal effect via Performer FAVOR+ Positive Orthogonal Random Features.

Standard Transformer self-attention computes an explicit softmax attention matrix
with quadratic :math:`\mathcal{O}(T^2)` time and memory complexity. By factoring
the exponential softmax kernel :math:`\exp(q^\top k / \sqrt{d})` through strictly
positive random feature maps :math:`\mathbb{E}[\phi(q)^\top \phi(k)]` (Choromanski
et al., 2021), causal attention reduces to a linear-time :math:`\mathcal{O}(T m d)`
prefix scan. The resulting attention representations form an explicit Primal basis
matrix :math:`\Phi_{\text{attn}}` whose parameters are estimated in closed form by
the penalized weighted least-squares solver.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_linear_recurrence, from_btk, to_btd


def _sample_orf_weights(
    m: int,
    d: int,
    seed: int = 42,
    orthogonal: bool = True,
    scaling: int = 1,
) -> torch.Tensor:
    r"""Samples random projection weights using Orthogonal Random Features (ORF) or i.i.d. Gaussian.

    For ORF (:math:`\text{orthogonal}=\text{True}`), standard Gaussian blocks in
    :math:`\mathbb{R}^{d \times d}` are orthonormalized via QR decomposition with Mezzadri
    diagonal phase correction, then scaled by :math:`\sqrt{d}` (or :math:`\chi_d` norms)
    to match the first two moments of the isotropic Gaussian distribution while achieving
    strictly lower kernel estimation variance.

    Args:
        m (int): Number of random feature projections (rows).
        d (int): Dimensionality of the input features (columns).
        seed (int): Deterministic PRNG seed.
        orthogonal (bool): Whether to enforce block-orthogonal projection rows.
        scaling (int): 1 for exact constant :math:`\sqrt{d}` scaling (recommended),
            or 0 for row-wise :math:`\chi_d` norms.

    Returns:
        torch.Tensor: Projection weight matrix :math:`\Omega` of shape ``(m, d)``
            in ``float64``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    if not orthogonal:
        return torch.randn(m, d, generator=gen, dtype=torch.float64)

    nb_full_blocks = int(m // d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured = torch.randn(d, d, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(unstructured)
        d_diag = torch.diag(r)
        ph = torch.sign(d_diag)
        ph[ph == 0.0] = 1.0
        q = q * ph.unsqueeze(0)
        block_list.append(q)

    remaining = m - nb_full_blocks * d
    if remaining > 0:
        unstructured = torch.randn(d, d, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(unstructured)
        d_diag = torch.diag(r)
        ph = torch.sign(d_diag)
        ph[ph == 0.0] = 1.0
        q = q * ph.unsqueeze(0)
        block_list.append(q[:remaining])

    final_q = torch.cat(block_list, dim=0)

    if scaling == 1:
        multiplier = np.sqrt(float(d)) * torch.ones(m, dtype=torch.float64)
    else:
        multiplier = torch.randn(m, d, generator=gen, dtype=torch.float64).norm(dim=1)

    return multiplier.unsqueeze(-1) * final_q


class TransformerEffect(BaseEffect):
    r"""Transformer attention effect linearized via FAVOR+ Positive Orthogonal Random Features.

    Constructs a causal (or non-causal) multi-head attention representation matrix
    :math:`\Phi_{\text{attn}}` in :math:`\mathcal{O}(T m d)` time using prefix-sum
    linear attention over strictly positive random features:

    .. math::

        \phi(x) = \frac{1}{\sqrt{m}} \exp\left( \Omega x - \frac{1}{2}\|x\|_2^2 \right)

    Args:
        feature_name (str): Primary input feature name.
        m_features (int): Number of positive random features :math:`m`. Defaults to 16.
        n_heads (int): Number of multi-scale attention heads :math:`H`. Defaults to 4.
        causal (bool): Whether attention is strictly causal (:math:`\tau \le t`). Defaults to True.
        decay (Union[float, Tuple[float, ...], List[float]]): Exponential memory decay
            factors :math:`\gamma_h \in (0, 1]` per head. Defaults to ``(1.0, 0.99, 0.95, 0.85)``.
        mode (str): Output basis mode: ``'head'`` concatenates :math:`H` attention context
            vectors (:math:`H \cdot d` coefficients), while ``'bilinear'`` outputs the full
            random-feature state tensor (:math:`m \cdot d` coefficients). Defaults to ``'head'``.
        orthogonal (bool): Whether to use Orthogonal Random Features (ORF) for variance reduction.
            Defaults to True.
        seed (int): Random seed for the random projection matrix. Defaults to 42.
        lambda_p (float): Ridge penalty multiplier. Defaults to 1.0.
        extrapolate (str): Extrapolation mode. Defaults to ``'continue'``.
        additional_features (Optional[List[str]]): Additional multivariate feature names.
        k (Optional[int]): Total coefficient budget. If provided, sets ``n_heads = max(1, k // d)``
            in ``'head'`` mode, or ``m_features = max(1, k // d)`` in ``'bilinear'`` mode.
    """

    def __init__(
        self,
        feature_name: str,
        m_features: int = 16,
        n_heads: int = 4,
        causal: bool = True,
        decay: Union[float, Tuple[float, ...], List[float], str] = (1.0, 0.99, 0.95, 0.85),
        mode: str = "head",
        orthogonal: bool = True,
        seed: int = 42,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
        k: Optional[int] = None,
    ):
        """Initializes the TransformerEffect."""
        super().__init__(feature_name, "transformer", lambda_p, extrapolate)

        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)

        d_in = len(self.input_features)
        self.mode = str(mode).lower().strip()
        if self.mode not in ("head", "bilinear"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'head' or 'bilinear'.")

        if k is not None and int(k) > 0:
            if self.mode == "head":
                self.n_heads = max(1, int(k) // d_in)
                self.m_features = int(m_features)
            else:
                self.m_features = max(1, int(k) // d_in)
                self.n_heads = int(n_heads)
        else:
            self.m_features = int(m_features)
            self.n_heads = int(n_heads)

        if self.m_features <= 0:
            raise ValueError(f"m_features must be positive, got {self.m_features}.")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}.")

        self.causal = bool(causal)
        self.orthogonal = bool(orthogonal)
        self.seed = int(seed)

        # Parse decay rates per head
        if isinstance(decay, str):
            clean = decay.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            decay_vals = [float(v.strip()) for v in clean.split(",") if v.strip()]
        elif isinstance(decay, (int, float)):
            decay_vals = [float(decay)]
        else:
            decay_vals = [float(v) for v in decay]

        if not decay_vals:
            decay_vals = [1.0]

        if len(decay_vals) < self.n_heads:
            last_val = decay_vals[-1]
            decay_vals = decay_vals + [last_val] * (self.n_heads - len(decay_vals))
        else:
            decay_vals = decay_vals[: self.n_heads]

        self.decay = tuple(decay_vals)

        # Cache for sampled projection weights: one independent set of m features per head, shape (H, m, d)
        self._cached_omega: Optional[torch.Tensor] = None
        self._cached_d: Optional[int] = None
        if self.stateful:
            self._check_temporal_extrapolate()

    @property
    def stateful(self) -> bool:
        """Causal attention carries its running memory between two passes; the bidirectional form is stateless."""
        return self.causal

    def initialize(self, x_col: torch.Tensor) -> None:
        """The projections are drawn from the seed: nothing is learned from the data."""
        return None

    def get_n_coeffs(self) -> int:
        """Returns the number of linear basis coefficients."""
        d_in = len(self.input_features)
        if self.mode == "head":
            return self.n_heads * d_in
        return self.m_features * d_in

    def summary_type(self) -> str:
        """Returns the summary type."""
        return "Transformer"

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return (
            f"m={self.m_features}, heads={self.n_heads}, mode='{self.mode}', "
            f"causal={self.causal}, orf={self.orthogonal}"
        )

    def _get_omega(self, d: int, is_orf: Optional[bool] = None) -> torch.Tensor:
        """Projection weights of every head, ``(H, m, d)``: one seeded draw of ``H * m`` rows, one block of ``m`` rows per head."""
        orf_flag = self.orthogonal if is_orf is None else bool(is_orf)
        if self._cached_omega is not None and self._cached_d == d and is_orf is None:
            return self._cached_omega

        weights = _sample_orf_weights(
            self.n_heads * self.m_features, d, seed=self.seed, orthogonal=orf_flag, scaling=1
        ).reshape(self.n_heads, self.m_features, d)
        if is_orf is None and d == len(self.input_features):
            self._cached_omega = weights
            self._cached_d = d
        return weights

    def _get_weights(self, d: int, is_orf: Optional[bool] = None, head: int = 0) -> torch.Tensor:
        """Projection weight matrix Omega of one head, shape (m, d)."""
        return self._get_omega(d, is_orf)[head]

    def _compute_prf(
        self,
        x: torch.Tensor,
        d: Optional[int] = None,
        is_orf: Optional[bool] = None,
        head: int = 0,
    ) -> torch.Tensor:
        r"""Computes Positive Random Feature map :math:`\phi(x)` of one head.

        .. math::

            \phi(x) = \frac{1}{\sqrt{m}} \exp\left( x \Omega^\top - \frac{1}{2}\|x\|_2^2 \right)

        Args:
            x (torch.Tensor): Scaled input tensor of shape ``(..., d)``.
            d (Optional[int]): Input dimension. If None, inferred from ``x.shape[-1]``.
            is_orf (Optional[bool]): Whether to use orthogonal random features.
            head (int): The head whose projection is used.

        Returns:
            torch.Tensor: Positive random features of shape ``(..., m)`` with values > 0.
        """
        dim_d = d if d is not None else x.shape[-1]
        weights = self._get_weights(dim_d, is_orf, head)
        w_aligned = weights.to(device=x.device, dtype=x.dtype)

        # Linear projection x @ Omega^T: (..., m)
        proj = x @ w_aligned.T

        # Squared norm: (..., 1)
        norm_sq = torch.sum(x * x, dim=-1, keepdim=True)

        arg = proj - 0.5 * norm_sq
        # Clamping avoids exponential overflow on extreme inputs
        arg = torch.clamp(arg, max=80.0)
        phi = torch.exp(arg) / np.sqrt(float(self.m_features))

        # Strictly positive lower bound to prevent division by zero in normalizer
        return torch.clamp(phi, min=1e-12)

    def _compute_attention_context(
        self,
        u_3d: torch.Tensor,
        initial_m: Optional[torch.Tensor] = None,
        initial_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Evaluates causal or non-causal linear self-attention over sequence, every head with its own projection.

        Args:
            u_3d (torch.Tensor): Input batch of shape ``(B, T, d)``.
            initial_m (Optional[torch.Tensor]): Initial running memory ``(B, H, m, d)``.
            initial_z (Optional[torch.Tensor]): Initial normalizer ``(B, H, m)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Output context feature map of shape ``(B, T, out_dim)``.
                - Terminal memory ``M_T`` of shape ``(B, H, m, d)``.
                - Terminal normalizer ``z_T`` of shape ``(B, H, m)``.
        """
        b_dim, t_len, d_in = u_3d.shape
        m = self.m_features
        h = self.n_heads

        # Rescale queries and keys by d^{1/4}
        scale_factor = float(d_in) ** 0.25
        u_scaled = u_3d / scale_factor
        v = u_3d  # Values
        eps = 1e-6

        head_outputs = []
        term_m_list = []
        term_z_list = []
        primary: Optional[torch.Tensor] = None

        for head_idx in range(h):
            # Positive random feature representation of this head: (B, T, m)
            phi = self._compute_prf(u_scaled, d=d_in, head=head_idx)

            if not self.causal:
                # Bidirectional full-sequence attention
                m_total = torch.einsum("btm,btd->bmd", phi, v)  # (B, m, d)
                z_total = torch.sum(phi, dim=1)  # (B, m)
                d_norm = torch.einsum("btm,bm->bt", phi, z_total).unsqueeze(-1) + eps  # (B, T, 1)
                c_h = torch.einsum("btm,bmd->btd", phi, m_total) / d_norm
                if head_idx == 0 and self.mode == "bilinear":
                    primary = torch.einsum("btm,bmd->btmd", phi / d_norm, m_total).reshape(b_dim, t_len, m * d_in)
                head_outputs.append(c_h)
                term_m_list.append(m_total)
                term_z_list.append(z_total)
                continue

            # Causal linear attention: two discounted running sums (features times values, and features), solved as linear recurrences along time
            gamma = float(self.decay[head_idx])
            init_m_h = None if initial_m is None else initial_m[:, head_idx].reshape(b_dim, m * d_in)
            init_z_h = None if initial_z is None else initial_z[:, head_idx]
            outer = (phi.unsqueeze(-1) * v.unsqueeze(-2)).reshape(b_dim, t_len, m * d_in)  # (B, T, m * d)
            m_seq = causal_linear_recurrence(outer, gamma, 1.0, init_m_h).reshape(b_dim, t_len, m, d_in)
            z_seq = causal_linear_recurrence(phi, gamma, 1.0, init_z_h)  # (B, T, m)

            num = torch.einsum("btm,btmd->btd", phi, m_seq)  # (B, T, d)
            d_norm = torch.sum(phi * z_seq, dim=-1, keepdim=True) + eps  # (B, T, 1)
            c_h = num / d_norm
            if head_idx == 0 and self.mode == "bilinear":
                # In bilinear mode, use the primary head's strictly causal running state sequence:
                # (phi_t / D_t) * M_t, shape (B, T, m, d) -> (B, T, m * d)
                primary = (phi.unsqueeze(-1) * m_seq / d_norm.unsqueeze(-1)).reshape(b_dim, t_len, m * d_in)
            head_outputs.append(c_h)
            term_m_list.append(m_seq[:, -1])
            term_z_list.append(z_seq[:, -1])

        terminal_m = torch.stack(term_m_list, dim=1)  # (B, H, m, d)
        terminal_z = torch.stack(term_z_list, dim=1)  # (B, H, m)

        out = torch.cat(head_outputs, dim=-1) if self.mode == "head" else primary  # (B, T, H * d) or (B, T, m * d)
        return out, terminal_m, terminal_z

#: <attn_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Builds explicit linearized attention Primal basis matrix: ``(B, T)`` or ``(B, T, d)`` in, ``(B, T, out)`` out."""
        x3, shape, layout = to_btd(x_col, self.multi_input)
        B, _, d_in = x3.shape
        if d_in != len(self.input_features):
            raise ValueError(f"attn expects {len(self.input_features)} input column(s), got {d_in}")
        init_m = init_z = None
        if self.causal and state_in is not None and "S" in state_in and "Z" in state_in:
            init_m = state_in["S"].to(device=x3.device, dtype=x3.dtype)
            init_z = state_in["Z"].to(device=x3.device, dtype=x3.dtype)
            expected = (B, self.n_heads, self.m_features, d_in)
            if tuple(init_m.shape) != expected or tuple(init_z.shape) != expected[:3]:
                raise ValueError(f"state 'S' has shape {tuple(init_m.shape)}, expected {expected}: continue_state needs the same groups and features as training")
        states, term_m, term_z = self._compute_attention_context(x3, initial_m=init_m, initial_z=init_z)
        if self.causal:
            self._state_out = {"S": term_m.detach().clone(), "Z": term_z.detach().clone()}
        return from_btk(states, shape, layout)
#: </attn_feature_map>

#: <attn_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Builds strictly positive-definite Ridge penalty matrix :math:`P_{\text{attn}}`."""
        d_in = len(self.input_features)

        if self.mode == "head":
            # Diagonal head decay penalty: (1 + h) (x) I_d
            h_idx = torch.arange(
                1, self.n_heads + 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
            )
            head_weights = torch.repeat_interleave(h_idx, d_in)
            diag = self.lambda_p * head_weights
            return torch.diag(diag)

        # Bilinear mode: (1 + ||\omega_j||^2) (x) I_d
        weights = self._get_weights(d_in)          # the first head's projection
        omega_norms_sq = torch.sum(weights * weights, dim=-1)
        rff_weights = torch.repeat_interleave(1.0 + omega_norms_sq, d_in)
        diag = self.lambda_p * rff_weights.to(
            device=TORCH_DEVICE, dtype=torch.get_default_dtype()
        )
        return torch.diag(diag)
#: </attn_penalty_matrix>
