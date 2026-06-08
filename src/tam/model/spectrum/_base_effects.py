"""Defines the base interface for all effects in the TAM spectrum.

This module contains the abstract class that any effect (linear, spline, physics, etc.)
must implement to be compatible with the primal solver. It includes the universal 
Out-of-Distribution (OOD) extrapolation router.
"""

from abc import ABC, abstractmethod
import torch

#: <class_def>
class BaseEffect(ABC):
    """Abstract base class representing a single additive effect.

    An "effect" is a transformation of an input variable (e.g., Temperature)
    into a Reproducing Kernel Hilbert Space (RKHS), associated with a
    specific penalty matrix.

    Attributes:
        feature_name (str): The name of the data column associated with this effect.
        effect_type (str): Internal type identifier.
        lambda_p (float): The regularization weight.
        extrapolate (str): How to handle Out-Of-Distribution (OOD) data beyond [-1, 1].
                           Options: 'continue', 'constant', 'linear'.
    """

    def __init__(self, feature_name: str, effect_type: str, lambda_p: float, extrapolate: str):
        """Initializes the base effect."""
        self.feature_name = feature_name
        self.effect_type = effect_type
        self.lambda_p = lambda_p
        self.extrapolate = str(extrapolate).replace("'", "").replace('"', '').strip().lower()
#: </class_def>

    def _align_device(self, x_data: torch.Tensor, *tensors: torch.Tensor):
        """
        Internal utility to ensure parameters match the input data device.
        Crucial for OOM-safe chunking fallbacks (GPU to CPU).
        """
        aligned = []
        for t in tensors:
            if t is not None and t.device != x_data.device:
                aligned.append(t.to(x_data.device))
            else:
                aligned.append(t)
        return aligned[0] if len(aligned) == 1 else tuple(aligned)

#: <universal_extrapolation>
    def transform(self, x_col: torch.Tensor) -> torch.Tensor:
        r"""
        The Universal Extrapolation Wrapper (Multidimensional).
        Intercepts normalized data [-1, 1]^d. If data is OOD, it projects the 
        feature map according to the requested extrapolation mode using directional derivatives.
        """
        if self.extrapolate == 'continue':
            return self.build_feature_map(x_col)

        # 1. Identify the boundary of the safe hypercube [-1, 1]^d
        k_min, k_max = -1.0, 1.0
        x_bound = torch.clamp(x_col, min=k_min, max=k_max)
        
        # 2. Determine if the effect processes multi-dimensional feature coordinates natively
        if self.__class__.__name__ == 'TreeEffect':
            # Dynamically read the tree's internal architecture!
            is_multivariate = self.is_oblivious_binary
        else:
            is_multivariate = self.__class__.__name__ in [
                'NeuralEffect', 'RBFEffect', 'TensorProductEffect', 'LinearTreeEffect'
            ]

        is_ood_mask = (x_col != x_bound)
        if is_multivariate:
            # Reduce across the feature dimension (last dimension)
            is_ood = is_ood_mask.any(dim=-1)
        else:
            # Every scalar is its own feature
            is_ood = is_ood_mask

        if not is_ood.any():
            return self.build_feature_map(x_col)

        # Constant Extrapolation (Plateau / Clamping)
        if self.extrapolate == 'constant':
            return self.build_feature_map(x_bound)

        # 3. Base feature map exactly at the boundary of the hypercube
        phi_base = self.build_feature_map(x_bound)

        # 4. Extract out-of-bound vectors. The boolean mask perfectly flattens 
        # all batch dimensions into a single N_ood dimension.
        x_ood_vals = x_col[is_ood]    # Univariate: [N_ood], Multivariate: [N_ood, d]
        x_bound_vals = x_bound[is_ood]# Univariate: [N_ood], Multivariate: [N_ood, d]
        
        delta = x_ood_vals - x_bound_vals
        eps = 1e-5
        
        # 5. Handle 1D vs N-D distance and unit vector properly
        if is_multivariate:
            dist = torch.norm(delta, dim=-1, keepdim=True)
            unit_delta = delta / dist
            dist_multiplier = dist
        else:
            dist = torch.abs(delta)
            unit_delta = torch.sign(delta)
            # Ensure broadcasting against phi_base[is_ood] which is [N_ood, D]
            dist_multiplier = dist.unsqueeze(-1) if dist.dim() < phi_base[is_ood].dim() else dist

        # 6. Step STRICTLY BACKWARD into the safe hypercube to evaluate the finite difference.
        step_back = eps * unit_delta
        phi_eps = self.build_feature_map(x_bound_vals - step_back)
        
        # The outward gradient (directional derivative) along the unit violation vector
        slope_unit = (phi_base[is_ood] - phi_eps) / eps

        # Linear Extrapolation (First-Order Taylor Expansion)
        if self.extrapolate == 'linear':
            phi_base[is_ood] = phi_base[is_ood] + slope_unit * dist_multiplier
            return phi_base

        # Saturation Extrapolation (Slow transition to a linear with slope at 0)
        if self.extrapolate == 'saturation':
            lam = 2.0
            damped_dist = (1.0 - torch.exp(-lam * dist_multiplier)) / lam
            phi_base[is_ood] = phi_base[is_ood] + slope_unit * damped_dist
            return phi_base

        raise ValueError(
            f"Unknown extrapolation mode: '{self.extrapolate}'. "
            "Use 'continue', 'constant', 'linear', or 'saturation'."
        )
#: </universal_extrapolation>


#: <abstract_methods>
    @abstractmethod
    def get_n_coeffs(self) -> int:
        """Returns the dimension of the feature space."""
        raise NotImplementedError

    @abstractmethod
    def build_feature_map(self, x_col: torch.Tensor) -> torch.Tensor:
        """Builds the feature map matrix.

        Args:
            x_col (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The partial feature map matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def build_penalty_matrix(self) -> torch.Tensor:
        """Builds the square penalty matrix."""
        raise NotImplementedError
#: </abstract_methods>