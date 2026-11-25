# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Multi-scale Wavelet Scattering Transform (WST) effect.

Cascades analytic complex Morlet wavelets, complex modulus non-linearities,
and low-pass scaling filters up to order m = 2 (Mallat 2012, Bruna & Mallat 2013).
Produces a deterministic deep representation with proved Lipschitz stability to
time-warping deformations and translation invariance.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
from scipy.fft import next_fast_len
import torch
import torch.nn.functional as F

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, from_btk, tail_state, to_btd


class ScatteringEffect(BaseEffect):
    r"""Wavelet Scattering Transform (WST) effect.

    Computes deterministic multi-order scattering coefficients:
    - Order 0: Low-pass local average :math:`S_0 x = x * \phi_J`
    - Order 1: Wavelet modulus envelopes :math:`S_1 x(\lambda_1) = |x * \psi_{\lambda_1}| * \phi_J`
    - Order 2: Modulation interactions :math:`S_2 x(\lambda_1, \lambda_2) = ||x * \psi_{\lambda_1}| * \psi_{\lambda_2}| * \phi_J`
      strictly for coarser scales :math:`\lambda_2 < \lambda_1`.

    Args:
        feature_name (str): Input covariate column name.
        J (int): Number of octaves. Defaults to 3.
        Q (int): Number of wavelets per octave. Defaults to 1.
        max_order (int): Maximum scattering order (1 or 2). Defaults to 2.
        causal (bool): Whether to enforce strictly causal convolutions. Defaults to True.
        order_pen (float): Base penalty penalty multiplier per order (:math:`\gamma_m = \text{order\_pen}^m`).
            Defaults to 4.0.
        xi_0 (float): Base carrier frequency of the mother Morlet wavelet. Defaults to :math:`0.85\pi`.
        lambda_p (float): Global regularization weight. Defaults to 1.0.
        extrapolate (str): Extrapolation policy. Defaults to 'continue'.
    """

    def __init__(
        self,
        feature_name: str,
        J: int = 3,
        Q: int = 1,
        max_order: int = 2,
        causal: bool = True,
        order_pen: float = 4.0,
        xi_0: float = 0.85 * math.pi,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
    ):
        super().__init__(feature_name, "wst", lambda_p, extrapolate)
        self.J = int(J)
        self.Q = int(Q)
        self.max_order = int(max_order)
        self.causal = bool(causal)
        self.order_pen = float(order_pen)
        self.xi_0 = float(xi_0)

        self.N1 = self.J * self.Q
        self.n_order2 = (self.N1 * (self.N1 - 1)) // 2 if self.max_order >= 2 else 0
        self._n_coeffs = 1 + self.N1 + self.n_order2

        self._max_filter_len = 0
        self._filter_cache: Dict[
            Tuple[str, str],
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
        ] = {}
        self._batched_filter_cache: Dict[
            Tuple[str, str],
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
        ] = {}
        self._order2_col: Dict[Tuple[int, int], int] = {}
        if self.max_order >= 2:
            col = 1 + self.N1
            for j1 in range(self.N1):
                for j2 in range(j1 + 1, self.N1):
                    self._order2_col[(j1, j2)] = col
                    col += 1

        self._build_filters()
        # Causal memory of the cascade: the low-pass filter, plus one wavelet per scattering order
        longest_psi = max(len(f) for f in self.psi_real)
        self._n_hist = (len(self.phi_filter) - 1) + self.max_order * (longest_psi - 1)
        if self.stateful:
            self._check_temporal_extrapolate()

    @property
    def stateful(self) -> bool:
        """The causal cascade carries the last samples between two passes; the symmetric form reads the whole pass."""
        return self.causal

    def initialize(self, x_col: torch.Tensor) -> None:
        """The wavelets and the low-pass filter are fixed by (J, Q, xi_0): nothing is learned from the data."""
        return None

    def get_n_coeffs(self) -> int:
        """Returns total number of scattering channels."""
        return self._n_coeffs

    def _build_filters(self) -> None:
        """Pre-constructs zero-mean admissible Morlet wavelets and low-pass scaling filter."""
        # Scales lambda_j = 2^(-j / Q) * xi_0
        self.scales = [self.xi_0 * (2.0 ** (-j / self.Q)) for j in range(self.N1)]

        # Low-pass filter scale 2^J
        sigma_low = 2.0 ** self.J
        half_l = max(8, int(math.ceil(3.0 * sigma_low)))
        if self.causal:
            t_low = torch.arange(0, 2 * half_l + 1, dtype=torch.float64)
            phi = torch.exp(-0.5 * (t_low / sigma_low) ** 2)
        else:
            t_low = torch.arange(-half_l, half_l + 1, dtype=torch.float64)
            phi = torch.exp(-0.5 * (t_low / sigma_low) ** 2)
        phi = phi / phi.sum()
        self.phi_filter = phi

        # Complex Morlet wavelets centered at tau = t - hw in [-hw, hw]
        # with the discrete admissibility correction (the wavelet has zero mean); see the math page of the effect.
        self.psi_real: List[torch.Tensor] = []
        self.psi_imag: List[torch.Tensor] = []

        max_len = len(self.phi_filter)

        for j in range(self.N1):
            xi_j = self.scales[j]
            sigma_j = max(1.0, 1.0 / xi_j)
            hw = max(6, int(math.ceil(3.5 * sigma_j)))
            tau = torch.arange(-hw, hw + 1, dtype=torch.float64)
            env = torch.exp(-0.5 * (tau / sigma_j) ** 2)
            env_sum = env.sum() + 1e-12

            real_raw = env * torch.cos(xi_j * tau)
            imag_raw = env * torch.sin(xi_j * tau)

            real_part = real_raw - (real_raw.sum() / env_sum) * env
            imag_part = imag_raw - (imag_raw.sum() / env_sum) * env

            # Normalize L1 norm
            norm = (real_part**2 + imag_part**2).sqrt().sum() + 1e-8
            real_part = real_part / norm
            imag_part = imag_part / norm

            # The filters are kept in float64; the zero-mean correction is redone in the dtype of the input (see _filters)
            self.psi_real.append(real_part)
            self.psi_imag.append(imag_part)
            max_len = max(max_len, len(real_part))

        self._max_filter_len = max_len

    def _filters(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """The filters in the dtype of the input; every wavelet keeps sum(psi) == 0 in that dtype."""
        key = (str(device), str(dtype))
        if key not in self._filter_cache:
            phi = self.phi_filter.to(device=device, dtype=dtype)
            re = [f.to(device=device, dtype=dtype) for f in self.psi_real]
            im = [f.to(device=device, dtype=dtype) for f in self.psi_imag]
            re = [f - f.mean() for f in re]
            im = [f - f.mean() for f in im]
            self._filter_cache[key] = (phi, re, im)
        return self._filter_cache[key]

    def _batched_filters(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Pre-packed batched wavelet filters for fast convolution in ``build_feature_map``."""
        key = (str(device), str(dtype))
        if key not in self._batched_filter_cache:
            phi, re, im = self._filters(device, dtype)
            max_psi = max(f.shape[0] for f in re) if self.N1 > 0 else 1
            psi_bank = torch.zeros(2 * self.N1, 1, max_psi, device=device, dtype=dtype)
            psi_pairs: List[torch.Tensor] = []
            for j in range(self.N1):
                k_j = re[j].shape[0]
                psi_pairs.append(torch.stack([re[j], im[j]], dim=0).unsqueeze(1))
                off = (max_psi - k_j) if self.causal else ((max_psi - k_j) // 2)
                psi_bank[2 * j, 0, off : off + k_j] = re[j]
                psi_bank[2 * j + 1, 0, off : off + k_j] = im[j]
            self._batched_filter_cache[key] = (phi, psi_bank, psi_pairs)
        return self._batched_filter_cache[key]

    def _causal_convolve(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Causal convolution of every row of ``x`` ``(B, L)`` with the FIR filter ``h``: y[t] depends on x[s], s <= t."""
        k_len = h.shape[0]
        h_weight = h.view(1, 1, k_len)
        # Causal left-pad by (k_len - 1) replicating the earliest available value of each row
        x_padded = F.pad(x.unsqueeze(1), (k_len - 1, 0), mode="replicate")
        return F.conv1d(x_padded, h_weight, padding=0).squeeze(1)

    def _symmetric_convolve(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Symmetric convolution of every row with same-length output."""
        k_len = h.shape[0]
        h_weight = h.view(1, 1, k_len)
        out = F.conv1d(x.unsqueeze(1), h_weight, padding=k_len // 2).squeeze(1)
        return out[:, : x.shape[1]]

    def _conv1d(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return self._causal_convolve(x, h)
        return self._symmetric_convolve(x, h)

    @staticmethod
    def _crop_causal_tail(v: torch.Tensor, n_steps: int) -> torch.Tensor:
        """The last ``n_steps`` columns of ``(B, n_hist + T)``: the part that belongs to the current chunk once the carried history is dropped."""
        return v[:, -n_steps:]

#: <wst_feature_map>
    def build_feature_map(self, x: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Computes scattering transform channels [S_0, S_1, S_2]: ``(B, T)`` in, ``(B, T, n_coeffs)`` out, each group alone."""
        x3, shape, layout = to_btd(x, False)
        if x3.shape[-1] != 1:
            raise ValueError(f"wst reads one column, got {x3.shape[-1]}")
        B, T, _ = x3.shape
        if T == 0:
            return from_btk(torch.zeros(B, 0, self._n_coeffs, dtype=x3.dtype, device=x3.device), shape, layout)
        phi_f, psi_bank, psi_pairs = self._batched_filters(x3.device, x3.dtype)

        if self.causal:
            x_full3 = causal_history(x3, self._n_hist, state_in)
            self._state_out = {"hist": tail_state(x_full3, self._n_hist)}
        else:
            x_full3 = x3
        x_full = x_full3[..., 0]                    # (B, L)
        L = x_full.shape[1]

        all_ch = torch.empty(B, self._n_coeffs, L, device=x3.device, dtype=x3.dtype)
        all_ch[:, 0, :] = x_full

        if self.N1 > 0:
            x_in1 = x_full.unsqueeze(1)
            m_len = psi_bank.shape[-1]
            if self.causal:
                w1 = F.conv1d(F.pad(x_in1, (m_len - 1, 0), mode="replicate"), psi_bank, padding=0)
            else:
                w1 = F.conv1d(x_in1, psi_bank, padding=m_len // 2)[..., :L]
            w1 = w1.view(B, self.N1, 2, L)
            u1 = torch.sqrt(w1[:, :, 0].square() + w1[:, :, 1].square() + 1e-10)
            all_ch[:, 1 : 1 + self.N1, :] = u1

            if self.max_order >= 2 and self.N1 >= 2:
                for j2 in range(1, self.N1):
                    pair_w = psi_pairs[j2]
                    k2 = pair_w.shape[-1]
                    u1_sub = u1[:, :j2, :].reshape(B * j2, 1, L)
                    if self.causal:
                        w2 = F.conv1d(F.pad(u1_sub, (k2 - 1, 0), mode="replicate"), pair_w, padding=0)
                    else:
                        w2 = F.conv1d(u1_sub, pair_w, padding=k2 // 2)[..., :L]
                    w2 = w2.view(B, j2, 2, L)
                    mod2 = torch.sqrt(w2[:, :, 0].square() + w2[:, :, 1].square() + 1e-10)
                    for j1 in range(j2):
                        all_ch[:, self._order2_col[(j1, j2)], :] = mod2[:, j1, :]

        k_phi = phi_f.shape[0]
        if self.causal:
            need = T + k_phi - 1
            inp_g = all_ch[:, :, L - need :] if L >= need else F.pad(all_ch, (need - L, 0), mode="replicate")
            out_slice_start = k_phi - 1
            pad_conv = 0
        else:
            inp_g = all_ch
            out_slice_start = k_phi // 2
            pad_conv = k_phi // 2

        if k_phi >= 32 and T >= 256:
            n_fft = int(next_fast_len(inp_g.shape[-1] + k_phi - 1, real=True))
            phi_spec = torch.fft.rfft(phi_f.flip(-1), n=n_fft).view(1, 1, -1)
            out = torch.fft.irfft(
                torch.fft.rfft(inp_g, n=n_fft, dim=-1) * phi_spec, n=n_fft, dim=-1
            )[..., out_slice_start : out_slice_start + T]
        else:
            phi_g = phi_f.view(1, 1, k_phi).expand(self._n_coeffs, 1, k_phi).contiguous()
            out = F.conv1d(inp_g, phi_g, padding=pad_conv, groups=self._n_coeffs)[..., :T]
        phi_cat = out.transpose(1, 2)
        return from_btk(phi_cat, shape, layout)
#: </wst_feature_map>

#: <wst_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Constructs multi-order Tikhonov scale penalty matrix."""
        penalties: List[float] = []

        # Order 0 penalty: gamma_0 = 1.0
        penalties.append(1.0)

        # Order 1 penalties: gamma_1 * (scale_ratio)^2
        gamma_1 = self.order_pen
        for j1 in range(self.N1):
            scale_ratio = self.scales[j1] / self.xi_0
            penalties.append(gamma_1 * (scale_ratio**2))

        # Order 2 penalties: gamma_2 * ((s1^2 + s2^2) / xi_0^2)
        if self.max_order >= 2:
            gamma_2 = self.order_pen**2
            for j1 in range(self.N1):
                s1 = self.scales[j1] / self.xi_0
                for j2 in range(j1 + 1, self.N1):
                    s2 = self.scales[j2] / self.xi_0
                    penalties.append(gamma_2 * (s1**2 + s2**2))

        dtype = torch.get_default_dtype()
        P = torch.diag(torch.tensor(penalties, device=TORCH_DEVICE, dtype=dtype))
        if self.N1 >= 3:
            # Neighbouring octaves of the order-1 envelopes vary smoothly: a second-difference coupling along the octaves
            D2 = torch.diff(torch.eye(self.N1, device=TORCH_DEVICE, dtype=dtype), n=2, dim=0)
            P[1 : 1 + self.N1, 1 : 1 + self.N1] += D2.T @ D2
        return self.lambda_p * P
#: </wst_penalty_matrix>
