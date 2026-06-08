r"""Implements Radial Basis Function (RBF) effects.

Supports Standard Gaussian kernels and Physics-Informed Matérn kernels.
"""

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
        extrapolate: str
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
        """
        kernel_type = f"rbf_{'matern' if nu else 'gauss'}"
        super().__init__(feature_name, kernel_type, lambda_p, extrapolate)
        self.n_centers = n_centers
        self.gamma = gamma 
        self.nu = nu
        self.centers = None 
        
        # --- Multivariate Handling (e.g., Latitude + Longitude) ---
        self.input_features = [feature_name]
        if additional_features:
            self.input_features.extend(additional_features)

    def get_n_coeffs(self) -> int:
        return self.n_centers

    def _init_params(self, x_in: torch.Tensor, is_dummy: bool = False):
        r"""Initializes centers (random sampling) and gamma (median heuristic)."""
        # Flatten batch dimension to sample from the entire dataset
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        n_samples = x_flat.shape[0]
        
        # --- Safeguard against the Memory Estimator ---
        if is_dummy:
            # Return mocked parameters for the VRAM footprint estimator without saving them
            mock_centers = torch.randn(self.n_centers, x_in.shape[-1], device=x_in.device)
            mock_gamma = self.gamma if self.gamma is not None else 1.0
            return mock_centers, mock_gamma

        #  Center Selection (Random Sampling)
        if n_samples <= self.n_centers:
            indices = torch.randint(0, n_samples, size=(self.n_centers,), device=x_in.device)
            self.centers = x_flat[indices]
        else:
            indices = torch.randperm(n_samples, device=TORCH_DEVICE)[:self.n_centers]
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
        return torch.eye(
            self.n_centers, device=TORCH_DEVICE, dtype=torch.get_default_dtype()
        ) * self.lambda_p
#: </penalty_matrix>