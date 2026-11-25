# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements the Prior Spectral Effect via Bochner's theorem and Quasi-Monte Carlo RFF.

Projects data onto an explicit primal feature map whose frequency measure approximates
the continuous spectral density of a Matérn, Lorentzian, or empirical time-series prior.
Regularization is governed by a diagonal Sobolev penalty matrix.
"""

from typing import Callable, List, Optional, Union
import math
import numpy as np
import torch
from scipy.stats import t

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect
from ._temporal_utils import eval_1d_dedup


def _van_der_corput_sequence(n: int, base: int) -> np.ndarray:
    """Generates a 1D Van der Corput sequence of length n for a given integer base."""
    seq = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        idx = i + 1
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


def _van_der_corput_matrix(n: int, d: int) -> np.ndarray:
    """Generates an (n, d) low-discrepancy matrix using prime bases."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    cols = []
    for j in range(d):
        base = primes[j % len(primes)]
        cols.append(_van_der_corput_sequence(n, base))
    return np.stack(cols, axis=1)


def _sample_qmc(n_samples: int, d: int, qmc_type: str = "halton", seed: int = 42) -> np.ndarray:
    """Samples low-discrepancy uniform points in (0, 1)^d."""
    qmc_type = str(qmc_type).lower().strip()
    if qmc_type in ("halton", "sobol"):
        try:
            from scipy.stats import qmc
            if qmc_type == "halton":
                sampler = qmc.Halton(d=d, seed=seed)
                u = sampler.random(n=n_samples)
            else:
                sampler = qmc.Sobol(d=d, seed=seed)
                u = sampler.random(n=n_samples)
        except (ImportError, Exception):
            u = _van_der_corput_matrix(n_samples, d)
    elif qmc_type in ("mc", "random"):
        rng = np.random.RandomState(seed)
        u = rng.uniform(0.0, 1.0, size=(n_samples, d))
    else:
        raise ValueError(f"Unknown qmc type '{qmc_type}'. Use 'halton', 'sobol', or 'mc'.")

    eps = 1e-5
    return np.clip(u, eps, 1.0 - eps)


class PriorSpectralEffect(BaseEffect):
    r"""Implements a Bochner prior spectral effect using Quasi-Monte Carlo Random Fourier Features.

    By Bochner's theorem, any continuous shift-invariant positive-definite kernel
    is the Fourier transform of a positive spectral measure. This effect samples
    frequencies from the analytical Matérn/Lorentzian spectral density or an
    empirical periodogram, and evaluates explicit cosine and sine projections with
    Sobolev smoothness regularization.

    Attributes:
        k (int): Number of frequencies M. The feature map dimension is 2 * k.
        prior (str or Callable): Spectral prior type ('matern', 'lorentzian', 'empirical', or callable).
        nu (float): Matérn smoothness parameter. nu=0.5 corresponds to the Lorentzian/Cauchy prior.
        length_scale (float): Characteristic length-scale parameter ell.
        sobolev_order (float): Sobolev penalty order s, penalizing modes by (1 + ||omega||^2)^s.
        qmc (str): Low-discrepancy sequence ('halton', 'sobol', or 'mc').
        seed (int): Deterministic sampling seed.
        frequencies (torch.Tensor): Sampled frequencies of shape (k, d).
    """

    def __init__(
        self,
        feature_name: str,
        k: int = 64,
        prior: Union[str, Callable] = "matern",
        nu: float = 1.5,
        length_scale: float = 1.0,
        sobolev_order: float = 1.0,
        qmc: str = "halton",
        seed: int = 42,
        lambda_p: float = 1.0,
        extrapolate: str = "continue",
        additional_features: Optional[List[str]] = None,
    ):
        """Initializes the PriorSpectralEffect."""
        if k <= 0:
            raise ValueError(f"Number of frequencies k must be positive, got {k}.")
        if length_scale <= 0.0:
            raise ValueError(f"length_scale must be positive, got {length_scale}.")
        if nu <= 0.0:
            raise ValueError(f"Smoothness nu must be positive, got {nu}.")

        super().__init__(feature_name, "prior_spectral", lambda_p, extrapolate)

        self.k = int(k)
        self.m_freqs = self.k
        self.prior = prior
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.sobolev_order = float(sobolev_order)
        self.qmc = str(qmc).lower().strip()
        self.seed = int(seed)

        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)
        self.keeps_feature_dim = len(self.input_features) > 1

        d = len(self.input_features)
        self.frequencies: Optional[torch.Tensor] = None
        self._is_initialized: bool = False
        self._sample_initial_frequencies(d)

    def _sample_initial_frequencies(self, d: int) -> None:
        """Draws initial frequencies if an analytical prior or callable is given."""
        if isinstance(self.prior, str) and self.prior.lower() == "empirical":
            # Initial placeholder frequencies before initialize() is called on the training data
            freqs = torch.arange(1, self.k + 1, dtype=torch.get_default_dtype()).unsqueeze(-1)
            if d > 1:
                freqs = freqs.repeat(1, d)
            self.frequencies = freqs
            return

        u = _sample_qmc(self.k, d, self.qmc, self.seed)

        if callable(self.prior):
            u_t = torch.tensor(u, dtype=torch.get_default_dtype())
            res = self.prior(u_t)
            if not isinstance(res, torch.Tensor):
                res = torch.tensor(res, dtype=torch.get_default_dtype())
            if res.dim() == 1:
                res = res.unsqueeze(-1)
            self.frequencies = res
            return

        prior_str = str(self.prior).lower().strip()
        w = np.zeros_like(u)

        if prior_str == "lorentzian" or (prior_str == "matern" and np.isclose(self.nu, 0.5)):
            for j in range(d):
                w[:, j] = np.tan(np.pi * (u[:, j] - 0.5)) / self.length_scale
        elif prior_str == "matern":
            df = 2.0 * self.nu
            for j in range(d):
                w[:, j] = t.ppf(u[:, j], df=df) / self.length_scale
        else:
            raise ValueError(f"Unknown prior '{self.prior}'. Use 'matern', 'lorentzian', 'empirical', or a callable.")

        self.frequencies = torch.tensor(w, dtype=torch.get_default_dtype())

    def initialize(self, x_cols: torch.Tensor) -> None:
        """Extracts empirical spectral densities from the training tensor if prior='empirical'."""
        if not (isinstance(self.prior, str) and self.prior.lower() == "empirical"):
            return

        with torch.no_grad():
            x_cpu = x_cols.detach().cpu().to(torch.float64).numpy()
            d = len(self.required_features)

            # series[j]: (n_groups, T) signals of column j; a window never spans two groups, so every group has its own periodogram
            if x_cpu.ndim == 3:
                series = [x_cpu[:, :, j] for j in range(d)]
            elif x_cpu.ndim == 2 and not self.multi_input:
                series = [x_cpu]
            elif x_cpu.ndim == 1:
                series = [x_cpu[None, :]]
            else:
                series = [x_cpu.reshape(-1, d)[:, j][None, :] for j in range(d)]

            u = _sample_qmc(self.k, d, self.qmc, self.seed)
            w = np.zeros((self.k, d))

            for j in range(d):
                signal = series[j] - series[j].mean(axis=1, keepdims=True)      # (n_groups, T), centred per group
                n_pts = signal.shape[1]
                if n_pts < 2:
                    w[:, j] = np.arange(1, self.k + 1)
                    continue

                power = (np.abs(np.fft.rfft(signal, axis=1)) ** 2).mean(axis=0)  # (T // 2 + 1,), averaged over the groups
                power = power + 1e-12
                freqs_hz = np.fft.rfftfreq(n_pts, d=1.0)
                omega_grid = 2.0 * np.pi * freqs_hz

                p = power / np.sum(power)
                cdf = np.cumsum(p)
                cdf[-1] = 1.0

                w[:, j] = np.interp(u[:, j], cdf, omega_grid)

            self.frequencies = torch.tensor(w, dtype=torch.get_default_dtype(), device=x_cols.device)
            self._is_initialized = True

    def summary_details(self) -> str:
        """Returns details string for model summary."""
        return f"k={self.k}, prior='{self.prior}', nu={self.nu}, ell={self.length_scale}, Sobolev(s={self.sobolev_order})"

    def summary_type(self) -> str:
        """Returns the summary type."""
        return "PriorSpectral"

    def get_n_coeffs(self) -> int:
        """Returns total number of basis coefficients (2 * k)."""
        return 2 * self.m_freqs

#: <pspec_feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Builds explicit Cosine/Sine Random Fourier Feature map scaled by 1/sqrt(M)."""
        if (
            isinstance(self.prior, str)
            and self.prior.lower() == "empirical"
            and not self._is_initialized
        ):
            self.initialize(x_col)
        d = len(self.required_features)
        self.frequencies = self._align_device(x_col, self.frequencies)

        if self.multi_input:
            if x_col.shape[-1] != d:
                x_expanded = x_col.unsqueeze(-1)
            else:
                x_expanded = x_col
        else:
            if x_col.dim() == 1:
                x_expanded = x_col.unsqueeze(-1)
            elif x_col.dim() == 2:
                x_expanded = x_col.unsqueeze(-1)
            elif x_col.shape[-1] != d:
                x_expanded = x_col.unsqueeze(-1)
            else:
                x_expanded = x_col

        freqs = self.frequencies.to(device=x_col.device, dtype=x_col.dtype)
        scale = 1.0 / math.sqrt(self.m_freqs)
        if d == 1:
            freqs_t = freqs.T

            def _eval_1d(xv: torch.Tensor) -> torch.Tensor:
                theta = torch.matmul(xv.unsqueeze(-1), freqs_t)
                return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) * scale

            return eval_1d_dedup(x_expanded.squeeze(-1), _eval_1d)

        theta = torch.matmul(x_expanded, freqs.T)
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) * scale
#: </pspec_feature_map>

#: <pspec_penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds diagonal Sobolev penalty matrix (1 + ||omega_m||^2)^s."""
        freqs = self.frequencies.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype())
        squared_norms = torch.sum(freqs ** 2, dim=-1)
        gamma = (1.0 + squared_norms) ** self.sobolev_order
        diag = self.lambda_p * torch.cat([gamma, gamma])
        return torch.diag(diag)
#: </pspec_penalty_matrix>
