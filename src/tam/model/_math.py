# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France) et Sorbonne Université
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-License-Identifier: LGPL-3.0-or-later
# Authors : Yann Allioux, Nathan Doumèche

"""
Core Mathematical Solver Functions.

This module hosts the pure and generic mathematical functions that form the
core of the Time series Additive Model (TAM) solver. These functions are designed to
operate on PyTorch tensors, often in batches (e.g., per-group), handling
linear algebra operations, system solving, and scoring metrics.
"""

from typing import Dict, List, Union, Callable, Optional
from collections import Counter
import torch
from tam.common.utils import TORCH_DEVICE
from tam.common.hardware import hw

# Import abstract classes for type hinting
from .spectrum import BaseEffect, OffsetEffect

#: <weighted_cov>
def _compute_weighted_covariances(
    phi: torch.Tensor,
    y_data: torch.Tensor,
    loss_L_star_L: torch.Tensor
) -> tuple:
    r"""
    Computes the loss-weighted covariance matrices (LHS and RHS of normal equations).

    Calculates:
    - cov_X = Phi.H @ (L*L) @ Phi  (Weighted Feature Covariance)
    - cov_XY = Phi.H @ (L*L) @ Y   (Weighted Feature-Target Covariance)

    Args:
        phi: The design matrix. Shape: (..., n_samples, n_coeffs).
        y_data: The target tensor. Shape: (..., n_samples, d_out).
        loss_L_star_L: The loss-weighting matrix. Shape: (d_out, d_out).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cov_X, cov_XY)
    """
    # Ensure all tensors are on the same device
    phi = phi.to(TORCH_DEVICE)
    y_data = y_data.to(TORCH_DEVICE)
    loss_L_star_L = loss_L_star_L.to(TORCH_DEVICE)
    
    # Cast y_data to match phi's dtype
    y_data_aligned = y_data.to(phi.dtype)

    #  Compute Weighted Y
    # y_weighted shape: (..., n_samples, d_out)
    y_weighted = y_data_aligned @ loss_L_star_L
    
    #  Compute RHS: cov_XY = Phi^H @ Y_weighted
    cov_XY = phi.mT @ y_weighted

    #  Compute LHS: cov_X
    # Note: Current implementation treats single-target and multi-target differently
    if loss_L_star_L.shape[0] == 1:
        # Single-target: Apply scalar weight via square root
        L_sqrt = loss_L_star_L[0, 0].sqrt()
        phi_weighted = phi * L_sqrt
        cov_X = phi_weighted.mT @ phi_weighted
    else:
        # Multi-target: Currently defaulting to unweighted feature covariance
        # (Phi^H @ Phi) for stability in hierarchical cases.
        cov_X = phi.mT @ phi

    return cov_X, cov_XY
#: </weighted_cov>

#: <solve_system>
def solve_linear_system(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    penalty_M_star_M: torch.Tensor,
    n_samples: Union[int, float]
) -> torch.Tensor:
    r"""
    Solves the regularized linear system (batched).
    
    Solves for Coefficients (Beta):
    (cov_X + n_samples * P) @ Beta = cov_XY

    Args:
        cov_X: The weighted feature covariance matrix. Shape: (..., K, K).
        cov_XY: The weighted feature-target covariance. Shape: (..., K, d_out).
        penalty_M_star_M: The penalty matrix P. Shape: (K, K).
        n_samples: Scaling factor for the penalty term (usually sample count).

    Returns:
        torch.Tensor: Fitted coefficients. Shape: (..., K, d_out).
    """
    cov_X = cov_X.to(TORCH_DEVICE)
    cov_XY = cov_XY.to(TORCH_DEVICE)
    penalty_M_star_M = penalty_M_star_M.to(TORCH_DEVICE)

    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()

    # Expand penalty matrix dimensions to match batch size of cov_X
    dims_to_add = cov_X.dim() - penalty_M_star_M.dim()
    regularization_term = (n_samples * penalty_M_star_M).view(
        *([1] * dims_to_add), *penalty_M_star_M.shape
    )
    
    # LHS construction
    matrix_to_invert = cov_X + regularization_term
    
    # Add Jitter for numerical stability (regularize diagonal)
    jitter_scale = 1e-6 * n_samples
    jitter = jitter_scale * torch.eye(
        matrix_to_invert.shape[-1], 
        device=TORCH_DEVICE, 
        dtype=matrix_to_invert.dtype
    )
    matrix_to_invert += jitter.view(*([1] * dims_to_add), *jitter.shape)

    # Solve system: A^-1 @ B
    coeffs_list = hw.safe_solve(matrix_to_invert, cov_XY)

    return coeffs_list
#: </solve_system>


def _predict_from_coeffs(
    phi_matrix: torch.Tensor,
    adaptive_coeffs: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes predictions using the linear model equation.
    
    Y_pred = Phi @ Coefficients

    Args:
        phi_matrix: Design matrix.
        adaptive_coeffs: Fitted coefficients.

    Returns:
        torch.Tensor: Predictions.
    """
    phi_matrix = phi_matrix.to(TORCH_DEVICE)
    adaptive_coeffs = adaptive_coeffs.to(TORCH_DEVICE)
    
    return phi_matrix @ adaptive_coeffs

#: <decompose>
def _decompose_prediction_tensor(
    phi_matrix: torch.Tensor,
    adaptive_coeffs: torch.Tensor,
    effects_list: List[BaseEffect],
) -> Dict[str, torch.Tensor]:
    r"""
    Decomposes the total prediction into contributions from individual features.

    Iterates through the global design matrix and coefficient vector, slicing
    them according to the size of each effect to compute `Phi_j @ Beta_j`.

    Smart Naming:
        - Unique features keep their name (e.g., 'temp').
        - Collisions (e.g., 's(x)' and 'l(x)') are prefixed (e.g., 's_x', 'l_x').

    Args:
        phi_matrix: Global design matrix.
        adaptive_coeffs: Global fitted coefficients.
        effects_list: List of effects defining the model structure.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping feature names to their
        contribution tensors (real-valued).
    """
    phi_matrix = phi_matrix.to(TORCH_DEVICE)
    adaptive_coeffs = adaptive_coeffs.to(TORCH_DEVICE)

    decomposed_effects = {}
    
    # Detect naming collisions (same feature used in multiple effects)
    feature_names = [e.feature_name for e in effects_list if not isinstance(e, OffsetEffect)]
    name_counts = Counter(feature_names)
    
    # Map effect types to short prefixes for disambiguation
    type_map = {
        'linear': 'l', 'fourier': 'f', 'spline': 's', 'wavelet': 'w',
        'chebyshev': 'p', 'categorical_nominal': 'c', 'categorical_ordinal': 'c',
        'neural': 'n', 'rbf_gauss': 'rbf', 'rbf_matern': 'rbf',
        'tensor_product': 'te', 'phys_spline': 'phys', 
        'phys_fourier': 'phys', 'phys_neural': 'phys', 'offset': 'offset'
    }

    coeff_idx = 0
    for i, effect in enumerate(effects_list):
        n_coeffs = effect.get_n_coeffs()
        
        # Slice the global matrices
        phi_slice = phi_matrix[..., :, coeff_idx : coeff_idx + n_coeffs]
        coeffs_slice = adaptive_coeffs[..., coeff_idx : coeff_idx + n_coeffs, :]
        
        # Compute contribution: Phi_j @ Beta_j
        contribution = (phi_slice @ coeffs_slice).squeeze(-1) # Assumes d_out=1
        
        base_name = effect.feature_name
        
        if isinstance(effect, OffsetEffect):
            final_name = "offset"
        elif name_counts[base_name] > 1:
            # Collision detected: Apply prefix
            prefix = type_map.get(effect.effect_type, effect.effect_type)
            final_name = f"{prefix}_{base_name}"
        else:
            final_name = base_name
            
        decomposed_effects[final_name] = contribution
        coeff_idx += n_coeffs
        
    return decomposed_effects
#: </decompose>

#: <gcv_score>
def compute_gcv_score(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    Y_sq: torch.Tensor,
    penalty_M_star_M: torch.Tensor,
    lambda_p: float,
    n_samples: int,
    gamma: float = 1.0
) -> torch.Tensor:
    r"""
    Computes the Generalized Cross Validation (GCV) score for a given lambda_p.
    
    GCV Formula (approx):
        GCV(lambda_p) = (1/n * RSS) / (1 - gamma/n * Tr(S))^2
    
    Numerical Optimization:
    Instead of computing the trace of the full Smoothing matrix S (N x N),
    we utilize the cyclic property of the trace to compute it on the 
    inverted covariance matrix (K x K), where K << N.

    Args:
        cov_X: Weighted feature covariance (Phi'Phi). Shape (B, K, K).
        cov_XY: Feature-Target covariance (Phi'Y). Shape (B, K, 1).
        Y_sq: Ground truth Y. Shape (B, N, 1).
        penalty_M_star_M: Penalty structure P. Shape (K, K).
        lambda_p: The regularization strength scalar to evaluate.
        n_samples: Number of samples (n).
        gamma: Multiplier for the effective degrees of freedom to prevent 
               overfitting in autocorrelated data (typically 1.4 - 1.5).

    Returns:
        torch.Tensor: The mean GCV score across the batch (scalar).
    """
    batch_size = cov_X.shape[0]
    
    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()
        
    #  Construct the Regularized System: A = (Phi'Phi + lambda_p * n * P)
    reg_term = (n_samples * lambda_p * penalty_M_star_M).unsqueeze(0) # Broadcast to batch
    
    A = cov_X + reg_term
    
    # Add jitter for numerical stability during inversion
    jitter = 1e-6 * torch.eye(A.shape[-1], device=cov_X.device, dtype=cov_X.dtype)
    A = A + jitter
    
    #  Solve for Coefficients: Beta = A^-1 @ cov_XY
    try:
        coeffs = hw.safe_solve(A, cov_XY) # Shape: (B, K, 1)
    except RuntimeError:
        return torch.tensor(float('inf'), device=cov_X.device)
    
    #  Compute Numerator: MSE (Mean Squared Error) on Training set
    # Using expansion: ||Y - Phi B||^2 = Y'Y - 2 B' Phi'Y + B' Phi'Phi B
        
    # Cross term: 2 * real(Beta^H @ cov_XY)
    cross_term = 2 * torch.sum(coeffs * cov_XY, dim=1).squeeze(-1)
    
    # Quadratic term: Beta^H @ cov_X @ Beta
    quad_term = (coeffs.mT @ cov_X @ coeffs).squeeze(-1).squeeze(-1)
    
    # Residual Sum of Squares (RSS)
    rss = torch.abs(Y_sq - cross_term + quad_term)
    mse = rss / n_samples
    
    #  Compute Denominator: Effective Degrees of Freedom (Trace of S)
    # Tr(S) = Tr(cov_X @ A^-1)
    # Since K is small (<1000), direct inversion on GPU is efficient.
    A_inv = torch.linalg.inv(A) 
    
    # Batch matrix multiplication: H = cov_X @ A_inv
    H_matrix = cov_X @ A_inv
    
    # Compute trace per batch element
    trace_H = torch.einsum('bii->b', H_matrix)
    
    # Protect against division by zero if model interpolates perfectly
    denom_factor = 1.0 - (gamma * trace_H / n_samples)
    denom_factor = torch.clamp(denom_factor, min=1e-6)
    
    #  Final GCV Score
    gcv = mse / (denom_factor ** 2)
    
    return torch.mean(gcv)
#: </gcv_score>

#: <solve_sparse_cg>
def solve_sparse_cg(
    compute_Av: Callable[[torch.Tensor], torch.Tensor], 
    b: torch.Tensor, 
    x0: Optional[torch.Tensor] = None, 
    tol: float = 1e-4, 
    max_iter: int = 1000
) -> torch.Tensor:
    r"""
    Solves the linear system Ax = b using the matrix-free Conjugate Gradient method.
    
    This solver never explicitly constructs the dense covariance matrix A. Instead, 
    it relies on a closure `compute_Av` that evaluates the matrix-vector product 
    (A @ v) dynamically. This allows for the resolution of massive algorithmic 
    structures (like Random Forests) without exhausting VRAM.

    Args:
        compute_Av: A closure evaluating (A @ v) for a given vector v.
        b: The right-hand side target vector (\Phi^* Y).
        x0: Initial guess for the coefficients.
        tol: Tolerance threshold for the residual norm.
        max_iter: Maximum number of iterations before stopping.

    Returns:
        torch.Tensor: The optimized coefficient vector.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - compute_Av(x)
    p = r.clone()
    
    is_batched = b.dim() >= 3
    sum_dims = tuple(range(1, b.dim())) if is_batched else tuple(range(b.dim()))

    rsold = torch.sum(r * r, dim=sum_dims, keepdim=True)
    norm_b = torch.sqrt(torch.sum(b * b, dim=sum_dims, keepdim=True))
    norm_b = torch.clamp(norm_b, min=1e-12)

    for i in range(max_iter):
        Ap = compute_Av(p)
        p_Ap = torch.sum(p * Ap, dim=sum_dims, keepdim=True)
        
        valid_mask = p_Ap > 1e-12
        lambda_p = torch.zeros_like(rsold)
        lambda_p[valid_mask] = rsold[valid_mask] / p_Ap[valid_mask]
        
        x = x + lambda_p * p
        r = r - lambda_p * Ap
        
        rsnew = torch.sum(r * r, dim=sum_dims, keepdim=True)
        
        if torch.all(torch.sqrt(rsnew) / norm_b < tol):
            break
            
        beta = torch.zeros_like(rsnew)
        beta[valid_mask] = rsnew[valid_mask] / rsold[valid_mask]
        
        p = r + beta * p
        rsold = rsnew

    return x
#: </solve_sparse_cg>