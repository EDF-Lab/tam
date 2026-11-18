# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Core Mathematical Solver Functions.

This module hosts the pure and generic mathematical functions that form the
core of the Time series Additive Model (TAM) solver. These functions are designed to
operate on PyTorch tensors, often in batches (e.g., per-group), handling
linear algebra operations, system solving, and scoring metrics.
"""

from typing import Dict, List, Union, Callable, Optional, Sequence, Tuple
from collections import Counter
import logging
import numpy as np
import torch
from tam.common.utils import TORCH_DEVICE
from tam.common.hardware import hw

# Import abstract classes for type hinting
from .spectrum import BaseEffect, OffsetEffect

logger = logging.getLogger(__name__)

#: <weighted_cov>
def _compute_weighted_covariances(
    phi: torch.Tensor,
    y_data: torch.Tensor,
    loss_L_star_L: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None
) -> tuple:
    r"""
    Computes the loss-weighted covariance matrices (LHS and RHS of normal equations).

    Calculates:
    - cov_X = Phi.H @ W @ (L*L) @ Phi  (Weighted Feature Covariance)
    - cov_XY = Phi.H @ W @ (L*L) @ Y   (Weighted Feature-Target Covariance)

    loss_L_star_L weights the output channel (shape (d_out, d_out)); sample_weights is an optional
    per-observation weight w (shape (..., n_samples, 1)) applied by a stable sqrt(W) row-scaling of both
    Phi and Y - the atom of any reweighted / GLM fit. sample_weights=None reproduces the plain
    least-squares path bit-for-bit.

    Args:
        phi: The design matrix. Shape: (..., n_samples, n_coeffs).
        y_data: The target tensor. Shape: (..., n_samples, d_out).
        loss_L_star_L: The loss-weighting matrix. Shape: (d_out, d_out).
        sample_weights: Optional per-observation weights. Shape: (..., n_samples, 1). Non-negative.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cov_X, cov_XY)
    """
    # Ensure all tensors are on the same device
    phi = phi.to(TORCH_DEVICE)
    y_data = y_data.to(TORCH_DEVICE)
    loss_L_star_L = loss_L_star_L.to(TORCH_DEVICE)

    # Cast y_data to match phi's dtype
    y_data_aligned = y_data.to(phi.dtype)

    if sample_weights is not None:
        root_w = sample_weights.to(device=TORCH_DEVICE, dtype=phi.dtype).clamp_min(0).sqrt()
        phi = phi * root_w
        y_data_aligned = y_data_aligned * root_w

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
def _ridge_floor(
    n_samples: Union[int, float],
    size: int,
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> torch.Tensor:
    r"""
    The ridge added to every regularised system for numerical stability: ``1e-6 * n * I``.

    The covariances are sums over the n samples, so the effective system is
    ``(Phi'Phi / n) + P + 1e-6 * I``: a penalty below about 1e-6 (``ap < -6``) is dominated by this floor.
    The solver and the GCV score use this one helper, so the penalty GCV selects gives the same
    coefficients when the model is refitted with it.
    """
    return (1e-6 * n_samples) * torch.eye(size, device=device, dtype=dtype)


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
    
    # Ridge floor for numerical stability (shared with compute_gcv_score)
    jitter = _ridge_floor(n_samples, matrix_to_invert.shape[-1], matrix_to_invert.dtype, TORCH_DEVICE)
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
def decomposition_names(effects_list: List[BaseEffect]) -> List[str]:
    r"""
    Names the additive component contributed by each effect, aligned with ``effects_list``.

    Smart Naming:
        - The intercept is 'offset'.
        - Unique features keep their name (e.g., 'temp').
        - Collisions (e.g., 's(x)' and 'l(x)') are prefixed (e.g., 's_x', 'l_x').
        - A collision that survives prefixing (two tensor products over the same features)
          is suffixed with its occurrence index, so no contribution silently overwrites another.

    ``decompose_prediction`` emits one ``effect_<name>`` column per entry. Callers that need
    the column of a specific formula term use this list rather than guessing from a feature
    name, which cannot resolve tensor products or several bases on one feature.

    Args:
        effects_list: List of effects defining the model structure.

    Returns:
        List[str]: One component name per effect, in the same order.
    """
    feature_names = [e.feature_name for e in effects_list if not isinstance(e, OffsetEffect)]
    name_counts = Counter(feature_names)

    # Map effect types to short prefixes for disambiguation
    type_map = {
        'linear': 'l', 'fourier': 'f', 'spline': 's', 'wavelet': 'w',
        'chebyshev': 'p', 'categorical_nominal': 'c', 'categorical_ordinal': 'c', 'categorical_fourier': 'c', 'categorical_graph': 'c',
        'neural': 'n', 'rbf_gauss': 'rbf', 'rbf_matern': 'rbf', 'rbf_tps': 'rbf', 'tree': 't', 'linear_tree': 'lt', 'pid': 'pid',
        'tensor_product': 'te', 'phys_spline': 'phys',
        'patch_basis': 'patch_basis',
        'temporal_operator': 'temp_op',
        'warped_calendar': 'wcal',
        'macro_trend': 'hp_trend',
        'degree_day': 'hdd_cdd',
        'thermal_inertia': 'thermal',
        'inertia': 'inertia', 'inertia_thermal': 'inertia', 'inertia_rotational': 'inertia', 'inertia_behavioral': 'inertia',
        'kan': 'kan',
        'single_index': 'si',
        'signature': 'sig',
        'phys_fourier': 'phys', 'phys_neural': 'phys', 'offset': 'offset'
    }

    names = []
    occurrences = Counter()
    for effect in effects_list:
        if isinstance(effect, OffsetEffect):
            name = "offset"
        elif name_counts[effect.feature_name] > 1:
            prefix = type_map.get(effect.effect_type, effect.effect_type)
            name = f"{prefix}_{effect.feature_name}"
        else:
            name = effect.feature_name
        occurrences[name] += 1
        names.append(name if occurrences[name] == 1 else f"{name}_{occurrences[name]}")
    return names


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
    names = decomposition_names(effects_list)

    coeff_idx = 0
    for effect, final_name in zip(effects_list, names):
        n_coeffs = effect.get_n_coeffs()

        # Slice the global matrices
        phi_slice = phi_matrix[..., :, coeff_idx : coeff_idx + n_coeffs]
        coeffs_slice = adaptive_coeffs[..., coeff_idx : coeff_idx + n_coeffs, :]

        # Compute contribution: Phi_j @ Beta_j
        contribution = (phi_slice @ coeffs_slice).squeeze(-1) # Assumes d_out=1

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
    gamma: float = 1.0,
    trace_method: str = "exact",
    n_probes: int = 32,
    seed: int = 42
) -> torch.Tensor:
    r"""
    Computes the Generalized Cross Validation (GCV) score for a given lambda_p.
    
    GCV Formula (approx):
        GCV(lambda_p) = (1/n * RSS) / (1 - gamma/n * Tr(S))^2
    
    Numerical Optimization:
    Instead of computing the trace of the full Smoothing matrix S (N x N),
    we utilize the cyclic property of the trace to compute it on the 
    inverted covariance matrix (K x K), where K << N.
    When trace_method='hutchinson' (or trace_method='auto' with K > 2000), uses Hutchinson Rademacher stochastic
    trace estimation without explicitly inverting A.

    Args:
        cov_X: Weighted feature covariance (Phi'Phi). Shape (B, K, K).
        cov_XY: Feature-Target covariance (Phi'Y). Shape (B, K, 1).
        Y_sq: Sum of squared targets Y'Y, one scalar per group. Shape (B,).
        penalty_M_star_M: Penalty structure P. Shape (K, K).
        lambda_p: The regularization strength scalar to evaluate.
        n_samples: Number of samples (n).
        gamma: Multiplier for the effective degrees of freedom to prevent 
               overfitting in autocorrelated data (typically 1.4 - 1.5).
        trace_method: 'exact', 'hutchinson', or 'auto'.
        n_probes: Number of Rademacher probe vectors for Hutchinson trace estimator.
        seed: Deterministic seed for Rademacher probes.

    Returns:
        torch.Tensor: The mean GCV score across the batch (scalar).
    """
    batch_size = cov_X.shape[0]
    
    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()
        
    #  Construct the Regularized System: A = (Phi'Phi + lambda_p * n * P)
    reg_term = (n_samples * lambda_p * penalty_M_star_M).unsqueeze(0) # Broadcast to batch
    
    A = cov_X + reg_term
    
    # Same ridge floor as solve_linear_system, so the scored model is the one fit() would return
    A = A + _ridge_floor(n_samples, A.shape[-1], cov_X.dtype, cov_X.device)
    
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
    rss = Y_sq - cross_term + quad_term
    # The expansion can dip below 0 by rounding when the fit is near-exact; clamp instead of folding it with abs()
    min_rss = torch.min(rss)
    if min_rss < -1e-8 * torch.max(torch.abs(Y_sq)).clamp(min=1.0):
        logger.debug("GCV: negative RSS %.3e clamped to 0 (rounding in Y'Y - 2B'Phi'Y + B'Phi'PhiB).", min_rss.item())
    rss = torch.clamp(rss, min=0.0)
    mse = rss / n_samples
    
    K = cov_X.shape[-1]
    if trace_method == "hutchinson" or (trace_method == "auto" and K > 2000):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed))
        z_cpu = torch.randint(0, 2, (K, n_probes), generator=gen, dtype=cov_X.dtype) * 2.0 - 1.0
        # Rademacher probes
        Z = z_cpu.to(device=cov_X.device, dtype=cov_X.dtype).unsqueeze(0).expand(batch_size, K, n_probes)
        AZ = hw.safe_solve(A, Z)
        cov_AZ = cov_X @ AZ
        trace_H = torch.mean(torch.sum(Z * cov_AZ, dim=1), dim=-1)
        denom_factor = torch.clamp(1.0 - (gamma * trace_H / n_samples), min=1e-6)
        return torch.mean(mse / (denom_factor ** 2))
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


def compute_gcv_score_and_grad(
    cov_X: torch.Tensor,
    cov_XY: torch.Tensor,
    Y_sq: torch.Tensor,
    penalty_blocks: List[torch.Tensor],
    spans: List[ tuple ],
    alpha_ps: torch.Tensor,
    n_samples: int,
    gamma: float = 1.4
) -> tuple:
    r"""
    Computes the GCV score and its exact analytical gradient with respect to
    log10 smoothing coordinates alpha_j = log10(lambda_j) (Wood 2004/2011).

    For A(lambda) = cov_X + n * sum_j 10^{alpha_j} S_j + ridge_floor:
      - d(Beta)/d(lambda_j) = -n * A^{-1} S_j Beta
      - d(RSS)/d(lambda_j) = 2 * n^2 * lambda_j * Beta^T P S_j A^{-1} cov_X Beta
      - d(Tr(S))/d(lambda_j) = -n * Tr(A^{-1} cov_X A^{-1} S_j)
      - d/d(alpha_j) = (ln 10) * lambda_j * d/d(lambda_j)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (scalar_gcv, grad_alpha of shape (J,))
    """
    dtype = cov_X.dtype
    device = cov_X.device
    B, K, _ = cov_X.shape
    J = len(penalty_blocks)
    ln10 = float(torch.log(torch.tensor(10.0, dtype=dtype)))

    lambdas = (10.0 ** alpha_ps.to(device=device, dtype=dtype))
    P_total = torch.zeros((K, K), dtype=dtype, device=device)
    S_blocks_full = []
    for j in range(J):
        s_start, s_end = spans[j]
        S_j = penalty_blocks[j].to(device=device, dtype=dtype)
        if S_j.is_sparse:
            S_j = S_j.to_dense()
        P_total[s_start:s_end, s_start:s_end] += lambdas[j] * S_j
        S_full = torch.zeros((K, K), dtype=dtype, device=device)
        S_full[s_start:s_end, s_start:s_end] = S_j
        S_blocks_full.append(S_full)

    A = cov_X + (n_samples * P_total).unsqueeze(0) + _ridge_floor(n_samples, K, dtype, device)
    A_inv = torch.linalg.inv(A)
    coeffs = A_inv @ cov_XY

    cross_term = 2.0 * torch.sum(coeffs * cov_XY, dim=1).squeeze(-1)
    quad_term = (coeffs.mT @ cov_X @ coeffs).squeeze(-1).squeeze(-1)
    rss = torch.clamp(Y_sq - cross_term + quad_term, min=1e-12)
    mse = rss / n_samples

    H_matrix = cov_X @ A_inv
    trace_H = torch.einsum('bii->b', H_matrix)
    denom = torch.clamp(1.0 - (gamma * trace_H / n_samples), min=1e-6)
    gcv_per_group = mse / (denom ** 2)
    gcv_mean = torch.mean(gcv_per_group)

    # Residual of the normal equations (without the penalty term), used by the derivative of the residual sum of squares
    r_cov = (cov_X @ coeffs) - cov_XY  # (B, K, 1)
    # Precompute A_inv @ cov_X @ A_inv for trace derivative
    Ainv_covX_Ainv = A_inv @ H_matrix  # (B, K, K)

    grad_alpha = torch.zeros(J, dtype=dtype, device=device)
    for j in range(J):
        s_start, s_end = spans[j]
        S_j = penalty_blocks[j].to(device=device, dtype=dtype)
        if S_j.is_sparse:
            S_j = S_j.to_dense()
        # d(Beta)/d(lambda_j) = -n * A_inv[:, :, s_start:s_end] @ (S_j @ coeffs[:, s_start:s_end, :])
        Sj_beta = S_j.unsqueeze(0) @ coeffs[:, s_start:s_end, :]  # (B, k_j, 1)
        dBeta_dlam = -float(n_samples) * (A_inv[:, :, s_start:s_end] @ Sj_beta)  # (B, K, 1)
        dRSS_dlam = 2.0 * torch.sum(r_cov * dBeta_dlam, dim=1).squeeze(-1)  # (B,)

        # d(Tr(S))/d(lambda_j) = -n * Tr(Ainv_covX_Ainv[:, s_start:s_end, s_start:s_end] @ S_j)
        sub_mat = Ainv_covX_Ainv[:, s_start:s_end, s_start:s_end] @ S_j.unsqueeze(0)
        dTr_dlam = -float(n_samples) * torch.einsum('bii->b', sub_mat)

        # Quotient rule on GCV_b = (RSS_b / n) / denom_b^2
        dMSE_dlam = dRSS_dlam / float(n_samples)
        ddenom_dlam = -(gamma / float(n_samples)) * dTr_dlam
        dGCV_dlam = (dMSE_dlam * denom - 2.0 * mse * ddenom_dlam) / (denom ** 3)
        grad_alpha[j] = torch.mean(dGCV_dlam) * lambdas[j] * ln10

    return gcv_mean, grad_alpha
#: </gcv_score>

#: <solve_sparse_cg>
def solve_sparse_cg(
    compute_Av: Callable[[torch.Tensor], torch.Tensor], 
    b: torch.Tensor, 
    x0: Optional[torch.Tensor] = None, 
    tol: float = 1e-4, 
    max_iter: int = 1000,
    preconditioner: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None
) -> torch.Tensor:
    r"""
    Solves the linear system Ax = b using the matrix-free (Preconditioned) Conjugate Gradient method.
    
    This solver never explicitly constructs the dense covariance matrix A. Instead, 
    it relies on a closure `compute_Av` that evaluates the matrix-vector product 
    (A @ v) dynamically. Supports optional Jacobi (inverse-diagonal tensor) or
    Block-Jacobi (callable closure M^{-1} r) preconditioning to accelerate convergence
    on multi-scale penalized designs.

    Args:
        compute_Av: A closure evaluating (A @ v) for a given vector v.
        b: The right-hand side target vector (\Phi^* Y).
        x0: Initial guess for the coefficients.
        tol: Tolerance threshold for the residual norm.
        max_iter: Maximum number of iterations before stopping.
        preconditioner: Optional inverse-diagonal tensor M_inv (Jacobi) or callable M_inv(r) (Block-Jacobi).

    Returns:
        torch.Tensor: The optimized coefficient vector.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    def apply_precond(vec: torch.Tensor) -> torch.Tensor:
        """Applies the preconditioner to a vector or to a block of vectors."""
        if preconditioner is None:
            return vec
        if callable(preconditioner):
            return preconditioner(vec)
        return preconditioner * vec

    r = b - compute_Av(x)
    z = apply_precond(r)
    p = z.clone()
    
    is_batched = b.dim() >= 3
    sum_dims = tuple(range(1, b.dim())) if is_batched else tuple(range(b.dim()))

    rz_old = torch.sum(r * z, dim=sum_dims, keepdim=True)
    norm_b = torch.sqrt(torch.sum(b * b, dim=sum_dims, keepdim=True))
    norm_b = torch.clamp(norm_b, min=1e-12)

    for _ in range(max_iter):
        Ap = compute_Av(p)
        p_Ap = torch.sum(p * Ap, dim=sum_dims, keepdim=True)
        
        valid_mask = p_Ap > 1e-12
        alpha_cg = torch.zeros_like(rz_old)
        alpha_cg[valid_mask] = rz_old[valid_mask] / p_Ap[valid_mask]
        
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        
        r_norm_sq = torch.sum(r * r, dim=sum_dims, keepdim=True)
        if torch.all(torch.sqrt(r_norm_sq) / norm_b < tol):
            break
            
        z = apply_precond(r)
        rz_new = torch.sum(r * z, dim=sum_dims, keepdim=True)

        beta = torch.zeros_like(rz_new)
        beta[valid_mask] = rz_new[valid_mask] / torch.clamp(rz_old[valid_mask], min=1e-18)
        
        p = z + beta * p
        rz_old = rz_new

    return x
#: </solve_sparse_cg>


#: <solve_block_qr>
def solve_block_qr(
    phi_chunks: Sequence[torch.Tensor],
    y_chunks: Sequence[torch.Tensor],
    penalty_M_star_M: torch.Tensor,
    n_samples: Optional[int] = None
) -> torch.Tensor:
    r"""
    Incremental Tall-Skinny Block-QR solver for penalized least squares.

    Avoids squaring the condition number \kappa(\Phi^T \Phi) = \kappa(\Phi)^2 by
    maintaining an upper-triangular factor R \in R^{K x K} across row chunks via
    reduced Householder QR updates, and appending the square-root penalty factor
    L_P (where n * P + ridge_floor = L_P^T L_P) as a final synthetic row block.

    Args:
        phi_chunks: Sequence of design matrix chunks, each of shape (B, N_c, K) or (N_c, K).
        y_chunks: Sequence of target chunks, each of shape (B, N_c, d_out) or (N_c, d_out).
        penalty_M_star_M: Penalty matrix P of shape (K, K).
        n_samples: Total sample count across chunks (inferred if None).

    Returns:
        torch.Tensor: Fitted coefficients of shape (B, K, d_out) or (K, d_out).
    """
    if not phi_chunks:
        raise ValueError("phi_chunks must contain at least one tensor chunk.")

    unbatched = (phi_chunks[0].dim() == 2)
    p_list = [p.unsqueeze(0) if p.dim() == 2 else p for p in phi_chunks]
    y_list = [y.unsqueeze(0) if y.dim() == 2 else y for y in y_chunks]

    B, _, K = p_list[0].shape
    d_out = y_list[0].shape[-1]
    dtype = p_list[0].dtype
    device = p_list[0].device

    total_n = int(sum(p.shape[1] for p in p_list)) if n_samples is None else int(n_samples)

    if penalty_M_star_M.is_sparse:
        penalty_M_star_M = penalty_M_star_M.to_dense()
    P_scaled = total_n * penalty_M_star_M.to(device=device, dtype=dtype) + _ridge_floor(total_n, K, dtype, device)
    # Symmetric eigendecomposition for exact square-root factor L_P: L_P^T @ L_P = P_scaled
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (P_scaled + P_scaled.mT))
    L_P = torch.diag(torch.sqrt(torch.clamp(eigvals, min=0.0))) @ eigvecs.mT  # (K, K)

    R_curr = torch.zeros((B, K, K), dtype=dtype, device=device)
    qty_curr = torch.zeros((B, K, d_out), dtype=dtype, device=device)

    for phi_c, y_c in zip(p_list, y_list):
        phi_c = phi_c.to(device=device, dtype=dtype)
        y_c = y_c.to(device=device, dtype=dtype)
        stacked_mat = torch.cat([R_curr, phi_c], dim=1)          # (B, K + N_c, K)
        stacked_rhs = torch.cat([qty_curr, y_c], dim=1)          # (B, K + N_c, d_out)
        Q_step, R_curr = torch.linalg.qr(stacked_mat, mode='reduced')
        qty_curr = Q_step.mT @ stacked_rhs

    # Append square-root penalty block [L_P, 0]
    L_P_batch = L_P.unsqueeze(0).expand(B, K, K)
    zero_rhs = torch.zeros((B, K, d_out), dtype=dtype, device=device)
    final_mat = torch.cat([R_curr, L_P_batch], dim=1)
    final_rhs = torch.cat([qty_curr, zero_rhs], dim=1)
    Q_fin, R_fin = torch.linalg.qr(final_mat, mode='reduced')
    qty_fin = Q_fin.mT @ final_rhs

    coeffs = hw.safe_solve(R_fin, qty_fin)
    return coeffs.squeeze(0) if unbatched else coeffs
#: </solve_block_qr>


#: <sqrt_group_lasso>
def solve_sqrt_group_lasso(
    phi: torch.Tensor,
    y: torch.Tensor,
    group_spans: List[tuple],
    c_scale: float = 1.1,
    alpha_level: float = 0.05,
    max_iter: int = 150,
    tol: float = 1e-5,
    penalty_matrix: Optional[torch.Tensor] = None,
    debiased_refit: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    r"""
    Tuning-free Pivotal Square-Root Group LASSO with debiased L2 Primal refit.

    Minimizes the pivotal self-normalizing objective (Belloni, Chernozhukov & Wang 2011/2014):
        min_theta  (1 / sqrt(T)) ||y - Phi theta||_2 + (lambda_0 / T) sum_g sqrt(p_g) ||theta_g||_2
    where lambda_0 = c_scale * sqrt(T) is independent of the unknown noise variance sigma^2.
    After identifying the active effect groups S_hat = {g : ||theta_g||_2 > 1e-12} (standardized solution), performs an
    exact debiased L2 Primal solve restricted to S_hat.

    Args:
        phi: Design matrix of shape (T, K) or (1, T, K).
        y: Target vector of shape (T, 1) or (1, T, 1).
        group_spans: List of (start_col, end_col) tuples for each additive effect group.
        c_scale: Pivotal multiplier c > 1 (default 1.1).
        alpha_level: Nominal test level in (0, 1) used to calibrate threshold.
        max_iter: Maximum proximal gradient iterations.
        tol: Relative parameter change convergence threshold.
        penalty_matrix: Optional K x K smoothness penalty matrix for the debiased L2 refit.
        debiased_refit: If True, refits active groups using exact penalized L2 normal equations.

    Returns:
        Tuple[torch.Tensor, List[int]]: (fitted_theta of shape matching input, active_group_indices)
    """
    was_3d = (phi.dim() == 3)
    X = phi.squeeze(0) if was_3d else phi
    Y = y.squeeze(0) if y.dim() == 3 else (y.unsqueeze(-1) if y.dim() == 1 else y)
    T, K = X.shape
    dtype = X.dtype
    device = X.device

    # Column-standardize X for pivotal group thresholding
    col_norms = torch.clamp(torch.linalg.norm(X, dim=0) / np.sqrt(float(T)), min=1e-8)
    X_std = X / col_norms.unsqueeze(0)

    # Estimate largest singular value squared L = ||X_std^T X_std||_2 / T via 10 power iterations
    v = torch.ones((K, 1), dtype=dtype, device=device) / np.sqrt(float(K))
    for _ in range(10):
        v_next = X_std.mT @ (X_std @ v)
        v = v_next / torch.clamp(torch.linalg.norm(v_next), min=1e-12)
    L_lip = float(torch.linalg.norm(X_std @ v).item() ** 2) / float(T)
    L_lip = max(L_lip, 1e-4)

    # Pivotal penalty level calibrated via Gaussian quantile approximation
    z_alpha = float(torch.erfinv(torch.tensor(1.0 - alpha_level / max(len(group_spans), 1), dtype=torch.float64)).item() * np.sqrt(2.0))
    base_lam = float(c_scale) * max(z_alpha, 1.0) / np.sqrt(float(T))

    theta = torch.zeros((K, 1), dtype=dtype, device=device)
    y_scale = max(float(torch.linalg.norm(Y).item()) / np.sqrt(float(T)), 1e-300)  # the floors below are relative to the units of the response
    for _ in range(max_iter):
        theta_old = theta.clone()
        residual = Y - (X_std @ theta)
        res_norm = float(torch.linalg.norm(residual).item()) / np.sqrt(float(T))
        if res_norm < 1e-10 * y_scale:
            break

        # Gradient of sqrt-RSS: -(X_std^T residual) / (T * res_norm)
        grad = -(X_std.mT @ residual) / (float(T) * res_norm)
        step = 1.0 / (L_lip / max(res_norm, 1e-4 * y_scale))
        u = theta - step * grad

        # Group-wise block soft-thresholding (Leave intercept / 1-column offset unpenalized if index 0 has size 1 and constant)
        theta_new = torch.zeros_like(u)
        for g_idx, (s0, s1) in enumerate(group_spans):
            p_g = s1 - s0
            u_g = u[s0:s1]
            # Do not zero out a pure intercept column
            if p_g == 1 and torch.allclose(X[:, s0], torch.ones_like(X[:, s0]), atol=1e-5):
                theta_new[s0:s1] = u_g
                continue
            thresh = step * base_lam * (1.0 + 0.25 * np.sqrt(float(p_g)))
            norm_ug = float(torch.linalg.norm(u_g).item())
            if norm_ug > thresh:
                theta_new[s0:s1] = (1.0 - thresh / norm_ug) * u_g

        theta = theta_new
        if float(torch.linalg.norm(theta - theta_old).item()) <= tol * max(float(torch.linalg.norm(theta_old).item()), 1.0):
            break

    theta_unscaled = theta / col_norms.unsqueeze(-1)
    # The block soft-thresholding sets an inactive group exactly to zero: read the active groups on the standardized solution,
    # where the test does not depend on the units of the response.
    active_groups = [
        g_idx for g_idx, (s0, s1) in enumerate(group_spans)
        if float(torch.linalg.norm(theta[s0:s1]).item()) > 1e-12
    ]
    if not active_groups:
        active_groups = [0]

    if debiased_refit:
        active_cols = []
        for g_idx in active_groups:
            s0, s1 = group_spans[g_idx]
            active_cols.extend(range(s0, s1))
        idx_t = torch.tensor(active_cols, dtype=torch.long, device=device)
        X_act = X[:, idx_t]
        cov_X_act = (X_act.mT @ X_act).unsqueeze(0)
        cov_XY_act = (X_act.mT @ Y).unsqueeze(0)
        if penalty_matrix is not None:
            P_dense = penalty_matrix.to_dense() if penalty_matrix.is_sparse else penalty_matrix
            P_act = P_dense.to(device=device, dtype=dtype)[idx_t][:, idx_t]
        else:
            P_act = torch.zeros((len(active_cols), len(active_cols)), dtype=dtype, device=device)
        beta_act = solve_linear_system(cov_X_act, cov_XY_act, P_act, T).squeeze(0)
        theta_final = torch.zeros((K, 1), dtype=dtype, device=device)
        theta_final[idx_t] = beta_act
    else:
        theta_final = theta_unscaled

    return (theta_final.unsqueeze(0) if was_3d else theta_final), active_groups


def compute_scale_stability_penalty(u_proj: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    r"""
    Computes the nested scale-stability variance penalty (Collarin et al., 2025):
        R_var(u) = (Var(u) - target_var)^2
    preventing scale indeterminacy between inner index projections and outer spline bases.
    """
    var_u = torch.var(u_proj, dim=-2 if u_proj.dim() >= 2 else 0, unbiased=False)
    return torch.mean((var_u - float(target_var)) ** 2)
#: </sqrt_group_lasso>
