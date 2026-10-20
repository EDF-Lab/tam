# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Implements Radial Basis Function (RBF) effects.

Supports Standard Gaussian kernels, Physics-Informed Matérn kernels and the thin-plate spline (``kernel='tps'``).
"""

import itertools
import math
import warnings
import torch
import numpy as np
try:
    from scipy.special import kv, gamma as sc_gamma 
except ImportError:
    kv, sc_gamma = None, None
from typing import List, Optional

from tam.common.utils import TORCH_DEVICE
from ._base_effects import BaseEffect

#: <init_rbf>
class RBFEffect(BaseEffect):
    r"""
    Implements RBF effects using Gaussian or Matérn kernels.
    
    Supports multivariate inputs (e.g., Spatial Latitude/Longitude) via 
    the `additional_features` argument.

    Attributes:
        n_centers (int): Number of anchor points (centers).
        gamma (float): Scale parameter (inverse length-scale squared).
        nu (float): Smoothness parameter for the Matérn kernel.
        input_features (List[str]): List of all input feature names consumed by this effect.
    """

    def __init__(
        self, 
        feature_name: str, 
        n_centers: int, 
        gamma: Optional[float], 
        nu: Optional[float],
        lambda_p: float,
        additional_features: Optional[List[str]],
        extrapolate: str,
        seed: int = 42,
        kernel: str = 'matern',
        m: int = 2,
        lambda_null: Optional[float] = None
        ):
        r"""
        Initializes the RBF Effect.

        Args:
            feature_name: Name of the primary feature.
            n_centers: Number of RBF centers to sample.
            gamma: Bandwidth parameter (1/2sigma^2). Auto-tuned if None.
            nu: Smoothness parameter for Matérn kernel. Uses Gaussian if None.
            lambda_p: Regularization strength (Ridge).
            additional_features: List of other feature names for multivariate kernels
                                 (e.g., ['Longitude'] if feature_name is 'Latitude').
            extrapolate (str): Strategy for Out-Of-Distribution (OOD) prediction beyond the 
                               [-1, 1] domain. Options: 'continue' (native topology), 
                               'constant' (hard clamp), 'linear' (first-order Taylor slope), 
                               or 'saturation' (smooth asymptotic clamp). Defaults to 'continue'.
            seed (int): Seed of the centre sampling (formula ``rbf(x, ..., seed=42)``). A local generator is used,
                        so the centres are reproducible and the global torch random state is untouched.
            kernel (str): ``'matern'`` (default: Matérn when ``nu`` is given, Gaussian otherwise) or ``'tps'``, the thin-plate spline:
                        parameter-free radial basis on the centres, bending-energy penalty, polynomial part of degree ``< m``.
            m (int): Order of the thin-plate spline (derivatives penalised), ``2m > d``. Only for ``kernel='tps'``.
            lambda_null (float, optional): Weight of the polynomial part of the thin-plate spline (a second smoothing parameter for
                        ``auto_fit``); a tiny fixed ridge when None. Only for ``kernel='tps'``.
        """
        kernel = str(kernel).lower().strip()
        if kernel not in ('matern', 'gauss', 'gaussian', 'tps'):
            raise ValueError(f"Unknown rbf kernel '{kernel}'. Use 'matern' (Matérn with nu, Gaussian without) or 'tps'.")
        self.kernel = 'tps' if kernel == 'tps' else 'matern'
        self.m = int(m)
        self.lambda_null = None if lambda_null is None else float(lambda_null)
        n_dimensions = 1 + len(additional_features or [])
        if self.kernel == 'tps' and 2 * self.m <= n_dimensions:
            raise ValueError(f"The thin-plate spline needs 2m > d; got m={self.m} for {n_dimensions} coordinates.")
        kernel_type = 'rbf_tps' if self.kernel == 'tps' else f"rbf_{'matern' if nu else 'gauss'}"
        super().__init__(feature_name, kernel_type, lambda_p, extrapolate)
        self.n_centers = n_centers
        self.gamma = gamma 
        self.nu = nu
        self.centers = None 
        self.seed = int(seed)
        
        # --- Multivariate Handling (e.g., Latitude + Longitude) ---
        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)

    def initialize(self, x_cols: torch.Tensor) -> None:
        r"""Samples the centres (and the median-heuristic gamma) from the full training tensor (..., n_features)."""
        if self.centers is None:
            x_in = x_cols if x_cols.dim() >= 3 else x_cols.unsqueeze(-1)
            if self.kernel == 'tps':
                self._init_tps(x_in)
            else:
                self._init_params(x_in, is_dummy=False)

    def get_n_coeffs(self) -> int:
        if self.kernel == 'tps':
            return self.n_centers - 1  # the constant of the polynomial part is the intercept of the model
        return self.n_centers

    # --- thin-plate spline (kernel='tps') -----------------------------------------------------------
    def _n_polynomial_terms(self) -> int:
        r"""Monomials of total degree below ``m`` in ``d`` coordinates (the constant included)."""
        d = len(self.input_features)
        return math.comb(d + self.m - 1, d)

    def _exponents(self, lowest_degree: int = 0) -> list:
        r"""Exponent vectors of the monomials of total degree ``lowest_degree .. m - 1``."""
        d = len(self.input_features)
        exponents = []
        for degree in range(lowest_degree, self.m):
            for combination in itertools.combinations_with_replacement(range(d), degree):
                exponent = [0] * d
                for index in combination:
                    exponent[index] += 1
                exponents.append(exponent)
        return exponents

    def _monomials(self, x: torch.Tensor, lowest_degree: int = 0) -> torch.Tensor:
        r"""Monomials of ``x`` (..., d), shape (..., n_terms)."""
        columns = []
        for exponent in self._exponents(lowest_degree):
            term = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
            for axis, power in enumerate(exponent):
                if power:
                    term = term * x[..., axis] ** power
            columns.append(term)
        return torch.stack(columns, dim=-1)

    def _eta(self, r: torch.Tensor) -> torch.Tensor:
        r"""The thin-plate radial basis (Wood, 2003): ``r^(2m-d) log r`` for even d, ``r^(2m-d)`` for odd d, with their constants."""
        d, m = len(self.input_features), self.m
        power = 2 * m - d
        if d % 2 == 0:
            constant = (-1.0) ** (m + 1 + d // 2) / (2.0 ** (2 * m - 1) * math.pi ** (d / 2.0) * math.factorial(m - 1) * math.factorial(m - d // 2))
            safe = r.clamp_min(1e-30)
            return constant * torch.where(r > 0, r ** power * torch.log(safe), torch.zeros_like(r))
        constant = math.gamma(d / 2.0 - m) / (2.0 ** (2 * m) * math.pi ** (d / 2.0) * math.factorial(m - 1))
        return constant * r ** power

    def _prepare_basis(self) -> None:
        r"""From the centres: the basis ``Z`` of the vectors orthogonal to the polynomials, and the bending energy ``Z' E Z``."""
        centres = self.centers.to(torch.get_default_dtype())
        n_terms = self._n_polynomial_terms()
        t_matrix = self._monomials(centres)
        q, _ = torch.linalg.qr(t_matrix, mode='complete')
        self._null_basis = q[:, n_terms:]
        energy = self._eta(torch.cdist(centres, centres))
        energy = self._null_basis.T @ energy @ self._null_basis
        energy = 0.5 * (energy + energy.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(energy)
        self._energy = (eigenvectors * eigenvalues.clamp_min(0.0)) @ eigenvectors.T  # negative round-off eigenvalues set to zero

    def _init_tps(self, x_in: torch.Tensor) -> None:
        r"""Picks distinct centres from the training tensor (every distinct point when there are fewer than ``n_centers``)."""
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        rng = torch.Generator(device=x_in.device)
        rng.manual_seed(self.seed)
        if x_flat.shape[0] > 200000:
            x_flat = x_flat[torch.randperm(x_flat.shape[0], generator=rng, device=x_in.device)[:200000]]
        distinct = torch.unique(x_flat, dim=0)
        if distinct.shape[0] <= self.n_centers:
            chosen = distinct
        else:
            chosen = distinct[torch.randperm(distinct.shape[0], generator=rng, device=x_in.device)[:self.n_centers]]
        if chosen.shape[0] <= self._n_polynomial_terms():
            raise ValueError(f"The thin-plate spline of '{self.feature_name}' needs more than {self._n_polynomial_terms()} distinct points; got {chosen.shape[0]}.")
        self.centers = chosen
        self.n_centers = int(chosen.shape[0])
        self._prepare_basis()

    def _tps_feature_map(self, x_in: torch.Tensor) -> torch.Tensor:
        r"""``[eta(|x - c_j|) Z, polynomials of degree 1 .. m-1]``, shape (..., n_centers - 1)."""
        centres = self._align_device(x_in, self.centers)
        z = self._align_device(x_in, self._null_basis)
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        radial = self._eta(torch.cdist(x_flat, centres.to(x_flat.dtype), p=2.0)) @ z.to(x_flat.dtype)
        parts = [radial]
        if self.m > 1:
            parts.append(self._monomials(x_flat, lowest_degree=1))
        phi = torch.cat(parts, dim=-1)
        return phi.reshape(*x_in.shape[:-1], phi.shape[-1])

    def penalty_coordinates(self) -> list:
        if self.kernel == 'tps' and self.lambda_null is not None:
            return [self.lambda_p, self.lambda_null]
        return [self.lambda_p]

    def set_penalty_coordinates(self, values) -> None:
        values = [float(v) for v in values]
        if len(values) != len(self.penalty_coordinates()):
            raise ValueError(f"rbf({self.feature_name}) has {len(self.penalty_coordinates())} penalty coordinates, got {len(values)}.")
        self.lambda_p = values[0]
        if len(values) == 2:
            self.lambda_null = values[1]

    def _init_params(self, x_in: torch.Tensor, is_dummy: bool = False):
        r"""Initializes centers (random sampling) and gamma (median heuristic)."""
        # Flatten batch dimension to sample from the entire dataset
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        n_samples = x_flat.shape[0]
        
        # Local generator: the centres depend on the seed only, and the global torch state is neither read nor advanced
        rng = torch.Generator(device=x_in.device)
        rng.manual_seed(self.seed)

        # --- Safeguard against the Memory Estimator ---
        if is_dummy:
            # Return mocked parameters for the VRAM footprint estimator without saving them
            mock_centers = torch.randn(self.n_centers, x_in.shape[-1], generator=rng, device=x_in.device)
            mock_gamma = self.gamma if self.gamma is not None else 1.0
            return mock_centers, mock_gamma

        #  Center Selection (Random Sampling)
        if n_samples <= self.n_centers:
            indices = torch.randint(0, n_samples, size=(self.n_centers,), generator=rng, device=x_in.device)
            self.centers = x_flat[indices]
        else:
            indices = torch.randperm(n_samples, generator=rng, device=x_in.device)[:self.n_centers]
            self.centers = x_flat[indices]

        #  Median Heuristic for Gamma (if not provided by user)
        if self.gamma is None:
            sub_size = min(1000, n_samples)
            sub_x = x_flat[:sub_size]
            dists = torch.cdist(sub_x, sub_x)
            median_dist = torch.median(dists)
            self.gamma = 1.0 / (median_dist ** 2) if median_dist > 0 else 1.0
            
        return self.centers, self.gamma
#: </init_rbf>

#: <matern_kernel>
    def _matern_kernel(self, dists: torch.Tensor, gamma_val: float) -> torch.Tensor:
        r"""
        Computes the Matérn kernel function.

        For machine learning, the smoothness parameter `nu` is typically set to half-integers
        (0.5, 1.5, 2.5). In these specific cases, the complex Modified Bessel function of the 
        second kind mathematically collapses into a simple product of an exponential and a 
        polynomial. This allows us to compute the kernel entirely on the GPU without relying 
        on CPU-bound Scipy numerical approximations.
        """
        # Pure PyTorch analytical solutions for standard ML Matern kernels (No CPU Sync!)
        if self.nu in [0.5, 1.5, 2.5]:
            scale = torch.sqrt(torch.tensor(gamma_val, device=dists.device))
            arg = np.sqrt(2 * self.nu) * dists * scale
            
            # Activate polynomial terms based on smoothness (nu)
            c1 = 1.0 if self.nu >= 1.5 else 0.0
            c2 = 1.0 / 3.0 if self.nu >= 2.5 else 0.0
            
            poly = 1.0 + c1 * arg + c2 * (arg ** 2)
            return (poly * torch.exp(-arg))
            
        # Fallback to Scipy for arbitrary fractional Nu (Forces CPU Sync)
        warnings.warn(f"Matérn with nu={self.nu} requires CPU synchronization. Use nu=0.5, 1.5, or 2.5 for pure GPU execution.")
        
        if kv is None:
            raise ImportError("scipy is required for arbitrary fractional Matern kernels.")

        # Transfer to CPU for Scipy operations
        d = dists.cpu().numpy()
        nu = self.nu
        scale_np = np.sqrt(gamma_val)
        
        # Avoid singularity at d=0 for numerical stability
        d[d == 0] = 1e-8
        
        sqrt_2_nu = np.sqrt(2 * nu)
        argument = sqrt_2_nu * d * scale_np
        
        # Matérn formula
        factor = (2 ** (1 - nu)) / sc_gamma(nu)
        term = (argument ** nu) * kv(nu, argument)
        
        result = factor * term
        result[d < 1e-7] = 1.0 
        
        return torch.tensor(result, device=dists.device, dtype=torch.get_default_dtype())
#: </matern_kernel>

#: <feature_map>
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        Builds the continuous Radial Basis Function (RBF) kernel feature map.
        
        Computes the Euclidean similarity vector between inputs and the fixed centers:
        Phi(x) = [k(x, c_1), ..., k(x, c_K)]
        
        Includes dynamic dimensional routing to guarantee compatibility with 
        multivariate Tensor Products (Kronecker te) and Out-Of-Distribution (OOD) Wrappers.
        """
        # 1. Architectural Shape Normalization
        if x_col.dim() == 1:
            x_in = x_col.unsqueeze(-1)
        elif x_col.dim() == 2:
            if x_col.shape[-1] == len(self.input_features):
                # Safeguard for the framework's (1, 1) VRAM memory probe
                if self.centers is None and x_col.shape == (1, 1) and len(self.input_features) == 1:
                    x_in = x_col.unsqueeze(-1)
                else:
                    x_in = x_col
            else:
                x_in = x_col.unsqueeze(-1)
        else:
            x_in = x_col

        # 2. Bulletproof Dummy Pass Detection (Memory Probe)
        # Prevents initialization triggers during VRAM footprint estimations
        is_dummy = False
        if self.centers is None:
            batch_shape = list(x_in.shape[:-1]) if x_in.dim() > 2 else list(x_in.shape)
            # Standardize [1] or [1,1] dummy signals
            if batch_shape == [1, 1] or batch_shape == [1]:
                is_dummy = True

        if self.kernel == 'tps':
            if self.centers is None:
                if is_dummy:
                    return torch.zeros(*x_in.shape[:-1], self.get_n_coeffs(), dtype=x_in.dtype, device=x_in.device)
                self._init_tps(x_in)
            return self._tps_feature_map(x_in)

        # 3. Lazy Initialization
        if self.centers is None:
            c, g = self._init_params(x_in, is_dummy)
        else:
            c, g = self.centers, self.gamma

        # Align centers with current chunk compute device
        c = self._align_device(x_in, c)

        # Flatten to (Total_Samples, D) for highly optimized torch.cdist compatibility
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        
        # Compute pairwise Euclidean distance natively on GPU
        dists_flat = torch.cdist(x_flat, c, p=2.0)
        
        # Reshape exactly back to the routed topological batch structure
        # Output Shape: (..., N_samples, N_centers)
        output_shape = list(x_in.shape[:-1]) + [c.shape[0]]
        dists = dists_flat.view(*output_shape)
        
        # 4. Apply the strictly positive Kernel Function
        if self.nu is not None:
            phi = self._matern_kernel(dists, g)
        else:
            # Standard Gaussian Kernel: exp(-gamma * d^2)
            phi = torch.exp(-g * (dists ** 2))
        
        return phi
#: </feature_map>

#: <penalty_matrix>
    def build_penalty_matrix(self) -> torch.Tensor:
        r"""
        Constructs the Ridge penalty matrix (Identity).
        
        Since RBF centers are isotropic, we penalize the magnitude 
        of coefficients uniformly.
        """
        if self.kernel == 'tps':
            n_polynomial = self._n_polynomial_terms() - 1
            if self.centers is None:
                return torch.eye(self.get_n_coeffs(), device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * self.lambda_p
            radial = self._energy.to(device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * self.lambda_p
            null_weight = self.lambda_null if self.lambda_null is not None else 1e-9  # a tiny fixed ridge keeps the system regular
            polynomial = torch.eye(n_polynomial, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * null_weight
            return torch.block_diag(radial, polynomial)
        return torch.eye(
            self.n_centers, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
        ) * self.lambda_p
#: </penalty_matrix>