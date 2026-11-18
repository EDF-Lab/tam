# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the Bispectrum Effect for higher-order spectral analysis and quadratic phase coupling.

For each frequency triad :math:`(\omega_1, \omega_2, \omega_3 = \omega_1 + \omega_2)` the effect reads the causal short-time Fourier
components of the series (a Hann window over the last ``n_fft`` samples) and forms their bispectral product
:math:`B(t) = z_1(t)\,z_2(t)\,\overline{z_3(t)}`, whose phase is constant when the three components are phase locked and
rotates otherwise (Hinich 1982; Brillinger and Rosenblatt 1967). Its real and imaginary parts, the modulus
:math:`|z_1 z_2|` and the cosine of the biphase :math:`\cos(\varphi_1 + \varphi_2 - \varphi_3)` are the four columns of the triad. Regularization is governed by a diagonal
Sobolev-bicoherence penalty matrix that penalizes high bi-frequencies and strongly shrinks spurious low-bicoherence
interactions.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import causal_history, from_btk, tail_state, to_btd


def _generate_default_triads(n_fft: int, k: int) -> List[Tuple[int, int]]:
    """Generates canonical default harmonic pairs (k1, k2) in the principal non-redundant domain.

    Domain: 1 <= k2 <= k1, k1 + k2 <= n_fft // 2.
    Sorted by sum (k1 + k2) then k1 to prioritize fundamental non-linear interactions.
    """
    max_k = n_fft // 2
    pairs = []
    for total_sum in range(2, max_k + 1):
        for k1 in range(1, total_sum):
            k2 = total_sum - k1
            if 1 <= k2 <= k1:
                pairs.append((k1, k2))
                if len(pairs) >= k:
                    return pairs
    return pairs[:k]


def _median_floor(v: torch.Tensor) -> torch.Tensor:
    """The median of all entries of ``v``, floored at 1e-12 (a scale that is never zero)."""
    return torch.clamp(v.reshape(-1).median(), min=1e-12)


class BispectrumEffect(BaseEffect):
    r"""Implements a higher-order polyspectral effect capturing quadratic phase coupling (QPC).

    For each frequency triad :math:`(k_1, k_2, k_3 = k_1 + k_2)` with :math:`\omega_i = 2\pi k_i / n_{\text{fft}}`, the causal
    demodulated components over the last ``n_fft`` samples are

    .. math::

        z_i(t) = \sum_{s=0}^{n_{\text{fft}}-1} h_s\, x(t-s)\, e^{-\mathrm{i}\omega_i (t-s)},

    with a Hann window :math:`h`. The phase factors of the bispectral product cancel when :math:`\omega_3 = \omega_1 + \omega_2`, so
    :math:`B(t) = z_1 z_2 \overline{z_3}` only depends on the samples inside the window. The four columns of the triad are
    :math:`\Re B / s`, :math:`\Im B / s`, :math:`|z_1 z_2| / s_a` and the cosine of the biphase
    :math:`\cos(\varphi_1 + \varphi_2 - \varphi_3) = \Re B / (|z_1 z_2|\,|z_3| + \varepsilon)`, with :math:`s, s_a` the training medians of
    :math:`|B|` and :math:`|z_1 z_2|` and :math:`\varepsilon = 10^{-12}` times the training median of :math:`|z_1 z_2|\,|z_3|`
    (all frozen when the effect is initialized). The four columns of a triad differ from those of every other triad.

    Regularization uses a bi-frequency Sobolev penalty weighted inversely by empirical bicoherence:
        P_{j} = \lambda_p \cdot \frac{1 + (\omega_{1, j}^2 + \omega_{2, j}^2 + \omega_{3, j}^2)^s}{\max(b^2_j, \varepsilon_b)} \otimes I_4

    Attributes:
        k (int): Number of coupled frequency triads. The feature map has 4 * k columns.
        n_fft (int): Length of the causal window (and of the discrete frequency grid).
        sobolev_order (float): Sobolev penalty order s.
        triads (List[Tuple[int, int]]): Discrete harmonic index pairs (k1, k2).
        bicoherences (List[float]): Bicoherence weights b^2 for each triad.
    """

    stateful = True

    def __init__(
        self,
        feature_name: str,
        k: int = 8,
        n_fft: int = 128,
        sobolev_order: float = 1.0,
        triads: Optional[Sequence[Any]] = None,
        bicoherences: Optional[Sequence[float]] = None,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
    ):
        super().__init__(feature_name, "bispectrum", lambda_p, extrapolate)

        self.k = int(k)
        self.n_fft = int(n_fft)
        self.s = float(sobolev_order)
        if additional_features:
            raise ValueError("bispec reads one column: the bispectrum is the phase coupling of one series with itself")
        self.additional_features = []
        self.input_features = [self.feature_name]
        self._n_hist = max(self.n_fft - 1, 0)

        parsed_triads: List[Tuple[int, int]] = []
        parsed_b2: List[float] = []

        if triads is not None and len(triads) > 0:
            for item in triads:
                if isinstance(item, dict):
                    k1 = int(item.get("k1", 1))
                    k2 = int(item.get("k2", 1))
                    b2 = float(item.get("bicoherence", 1.0))
                    parsed_triads.append((k1, k2))
                    parsed_b2.append(b2)
                elif isinstance(item, (tuple, list)):
                    if len(item) >= 3:
                        parsed_triads.append((int(item[0]), int(item[1])))
                        parsed_b2.append(float(item[2]))
                    elif len(item) == 2:
                        parsed_triads.append((int(item[0]), int(item[1])))
                        parsed_b2.append(1.0)
                    else:
                        raise ValueError(f"Invalid triad tuple: {item}")
                else:
                    raise ValueError(f"Unsupported triad format: {item}")

            self.triads = parsed_triads
            self.k = len(self.triads)
            if bicoherences is not None:
                self.bicoherences = [float(b) for b in bicoherences[: self.k]]
            else:
                self.bicoherences = parsed_b2
        else:
            self.triads = _generate_default_triads(self.n_fft, self.k)
            self.k = len(self.triads)
            if bicoherences is not None:
                self.bicoherences = [float(b) for b in bicoherences[: self.k]]
                self._auto_bicoherence = False
            else:
                self.bicoherences = [1.0] * self.k
                self._auto_bicoherence = True
        if triads is not None and len(triads) > 0:
            self._auto_bicoherence = False
        self._freqs = sorted({f for (k1, k2) in self.triads for f in (k1, k2, k1 + k2)})
        self._scales: Optional[torch.Tensor] = None   # (k, 3): medians of |B| and |z1 z2|, and the floor of the biphase denominator
        self._check_temporal_extrapolate()

    def initialize(self, x_col: torch.Tensor) -> None:
        """Freezes the scales of the columns and, when asked, the bicoherence weights, on the training tensor (every group)."""
        self._scales = None
        x3, _, _ = to_btd(x_col, False)
        if x3.numel() == 0:
            return
        if self.n_fft > x3.shape[1]:
            warnings.warn(
                f"bispec: the window ({self.n_fft} steps) is longer than the shortest group of the training data ({x3.shape[1]} steps): "
                "no window ever fills, so the columns come from zero-padded start-ups and do not describe the series.",
                UserWarning, stacklevel=2)
        if self._auto_bicoherence:
            self._estimate_bicoherence(x3[..., 0].detach().cpu().to(torch.float64).numpy())
        _, _, a, b, mag = self._components(causal_history(x3, self._n_hist, None), x3.shape[1])
        rows = []
        for j in range(self.k):
            median_bispectrum = _median_floor(mag[j])                                # scale of Re B and Im B
            median_pair_modulus = _median_floor(a[j])                                # scale of |z1 z2|
            biphase_floor = 1e-12 * (a[j] * b[j]).reshape(-1).median()               # keeps cos(biphase) finite where |z1 z2| |z3| is 0
            rows.append(torch.stack([median_bispectrum, median_pair_modulus, biphase_floor]))
        self._scales = torch.stack(rows).detach()                                     # (k, 3)

    def _estimate_bicoherence(self, series: np.ndarray) -> None:
        """Empirical bicoherence of each triad over the segments of every group (a segment never spans two groups)."""
        seg_len = min(self.n_fft, max(8, series.shape[1] // 2))
        f_segs = []
        for row in series:
            if len(row) < 16:
                continue
            x_centered = row - np.mean(row)
            for s_idx in range(max(1, len(row) // seg_len)):
                f_segs.append(np.fft.fft(x_centered[s_idx * seg_len : (s_idx + 1) * seg_len], n=self.n_fft))
        if not f_segs:
            return
        f_mat = np.stack(f_segs, axis=0)  # (n_segs, n_fft)
        new_b2 = []
        for k1, k2 in self.triads:
            k3 = (k1 + k2) % self.n_fft
            b_num = np.abs(np.mean(f_mat[:, k1] * f_mat[:, k2] * np.conj(f_mat[:, k3]))) ** 2
            b_den = (
                np.mean(np.abs(f_mat[:, k1] * f_mat[:, k2]) ** 2)
                * np.mean(np.abs(f_mat[:, k3]) ** 2)
                + 1e-12
            )
            new_b2.append(float(np.clip(b_num / b_den, 0.05, 1.0)))
        self.bicoherences = new_b2

    def summary_details(self) -> str:
        """Returns the summary description for StaticTAM.summary()."""
        return f"k={self.k}, Sobolev(s={self.s})"

    def get_n_coeffs(self) -> int:
        """Returns the number of primal coefficients (4 per triad)."""
        return 4 * self.k

    def _window_kernel(self, device, dtype) -> torch.Tensor:
        """Conv kernels ``(2F, 1, n_fft)``: real and imaginary parts of ``h_s exp(+i w s)`` per frequency, in correlation order."""
        n = self.n_fft
        s = torch.arange(n, dtype=torch.float64)
        hann = 0.5 - 0.5 * torch.cos(2.0 * math.pi * (s + 0.5) / n)
        rows = []
        for f in self._freqs:
            w = 2.0 * math.pi * float(f) / n
            rows.append(hann * torch.cos(w * s))
            rows.append(hann * torch.sin(w * s))
        kernel = torch.stack(rows, dim=0).flip(-1)       # kernel index j <-> lag s = n_fft - 1 - j
        return kernel.unsqueeze(1).to(device=device, dtype=dtype)

    def _components(self, x_hist: torch.Tensor, T: int):
        """Per-triad bispectral product and moduli of every step: lists over triads of ``(B, T)`` tensors."""
        B = x_hist.shape[0]
        out = F.conv1d(x_hist.transpose(1, 2), self._window_kernel(x_hist.device, x_hist.dtype))   # (B, 2F, T)
        y = {f: torch.complex(out[:, 2 * i], out[:, 2 * i + 1]) for i, f in enumerate(self._freqs)}
        re, im, a, b, mag = [], [], [], [], []
        for k1, k2 in self.triads:
            prod = y[k1] * y[k2]
            bis = prod * torch.conj(y[k1 + k2])
            re.append(bis.real)
            im.append(bis.imag)
            a.append(prod.abs())
            b.append(y[k1 + k2].abs())
            mag.append(bis.abs())
        return re, im, a, b, mag

#: <bispec_feature_map>
    def build_feature_map(self, x_col: torch.Tensor, state_in: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        r"""Builds the causal bispectral columns: ``(B, T)`` in, ``(B, T, 4 k)`` out; the state is the last ``n_fft - 1`` samples."""
        x3, shape, layout = to_btd(x_col, False)
        if x3.shape[-1] != 1:
            raise ValueError(f"bispec reads one column, got {x3.shape[-1]}")
        if self._scales is None:
            self.initialize(x_col)
        T = x3.shape[1]
        x_hist = causal_history(x3, self._n_hist, state_in)
        self._state_out = {"hist": tail_state(x_hist, self._n_hist)}
        re, im, a, b, _ = self._components(x_hist, T)
        scales = self._scales.to(device=x3.device, dtype=x3.dtype)
        cols = []
        for j in range(self.k):
            s_bis, s_a, eps = scales[j, 0], scales[j, 1], scales[j, 2]
            real_norm = re[j] / s_bis                                                # (B, T)
            imag_norm = im[j] / s_bis                                                # (B, T)
            pair_amp_norm = a[j] / s_a                                               # (B, T)
            cos_biphase = re[j] / (a[j] * b[j] + eps)                                # (B, T)
            cols.extend([real_norm, imag_norm, pair_amp_norm, cos_biphase])
        return from_btk(torch.stack(cols, dim=-1), shape, layout)
#: </bispec_feature_map>

#: <bispec_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""Constructs the diagonal Sobolev-bicoherence penalty matrix."""
        weights = []
        eps_b = 1e-4

        for j, (k1, k2) in enumerate(self.triads):
            k3 = k1 + k2
            omega1 = 2.0 * math.pi * float(k1) / self.n_fft
            omega2 = 2.0 * math.pi * float(k2) / self.n_fft
            omega3 = 2.0 * math.pi * float(k3) / self.n_fft

            sobolev_energy = 1.0 + (omega1**2 + omega2**2 + omega3**2) ** self.s
            b2 = max(self.bicoherences[j] if j < len(self.bicoherences) else 1.0, eps_b)

            triad_weight = self.lambda_p * (sobolev_energy / b2)
            # 4 identical diagonal weights for the 4 quadrature columns
            weights.extend([triad_weight] * 4)

        diag_tensor = torch.tensor(weights, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        return torch.diag(diag_tensor)
#: </bispec_penalty_matrix>

    def penalty_coordinates(self) -> list:
        """Returns the regularization weight lambda_p."""
        return [self.lambda_p]

    def set_penalty_coordinates(self, values: Sequence[float]) -> None:
        """Sets the regularization weight lambda_p."""
        vals = list(values)
        if len(vals) != 1:
            raise ValueError(f"BispectrumEffect expects 1 penalty coordinate, got {len(vals)}")
        self.lambda_p = float(vals[0])
