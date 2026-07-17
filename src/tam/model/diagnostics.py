# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Statistical Diagnostics Module for StaticTAM Models.

This module provides functions to perform statistical inference on a
trained StaticTAM model. It calculates the effective degrees of freedom,
noise variance (sigma^2), and t-statistics for all model coefficients.

The primary goal is to assess the statistical significance of each
feature (or "effect") in the model, even for complex, regularized
terms like splines and Fourier series.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from tam.common.utils import TORCH_DEVICE, _balance_groups
from tam.common.hardware import hw
from .additive import StaticTAM
from ._math import _predict_from_coeffs, _compute_weighted_covariances

# Import effect types for type checking and logic
from .spectrum import (
    BaseEffect, OffsetEffect, LinearEffect, FourierEffect, SplineEffect,
    CategoricalEffect, ChebyshevEffect, WaveletEffect,
    NeuralEffect, RBFEffect, UniversalPhysicsEffect
)

#: <p_matrix>
def _compute_p_matrix(
    phi: torch.Tensor, 
    penalty_M_star_M: torch.Tensor, 
    n_samples: int
) -> torch.Tensor:
    r"""Calculates the pseudo-inverse or coefficient solver matrix P.
    
    It maps the target variable Y to the fitted coefficients beta:
    beta_hat = P @ Y

    Args:
        phi (torch.Tensor): The batched design matrix.
             Shape: (n_groups, n_samples, n_coeffs)
        penalty_M_star_M (torch.Tensor): The global penalty matrix (unbatched).
             Shape: (n_coeffs, n_coeffs)
        n_samples (int): The number of samples (n) per group.

    Returns:
        torch.Tensor: The batched solver matrix P.
        Shape: (n_groups, n_coeffs, n_samples)
    """
    device = phi.device

    # Scale the unbatched penalty matrix and add a batch dimension
    # Explicitly move penalty matrix to the correct device
    regularization = (n_samples * penalty_M_star_M.to(device)).unsqueeze(0)
    
    # Left-hand side: (Phi'Phi + n * M*M)
    # Shape: (n_groups, n_coeffs, n_coeffs)
    matrix_to_invert = phi.mT @ phi + regularization
    
    # Add numerical jitter for stability on the targeted device
    jitter = 1e-6 * torch.eye(
        matrix_to_invert.shape[-1], 
        device=device, dtype=matrix_to_invert.dtype
    )
    matrix_to_invert += jitter.unsqueeze(0) 
    
    # Solve (LHS @ P = Phi') for P using the hardware abstraction layer
    return hw.safe_solve(matrix_to_invert, phi.mT)

def _estimate_noise_variance(
    phi: torch.Tensor, 
    p_matrix: torch.Tensor, 
    y_data: torch.Tensor, 
    predictions: torch.Tensor, 
    n_samples: int
) -> torch.Tensor:
    r"""Estimates the noise variance (sigma^2) for each group.

    Uses the unbiased estimator formula for regularized models.

    Args:
        phi (torch.Tensor): The batched design matrix.
        p_matrix (torch.Tensor): The batched solver matrix from `_compute_p_matrix`.
        y_data (torch.Tensor): The batched target variable.
        predictions (torch.Tensor): The batched model predictions (Y_hat).
        n_samples (int): The number of samples (n) per group.

    Returns:
        torch.Tensor: The estimated noise variance for each group.
        Shape: (n_groups,)
    """
    # Calculate the Hat matrix trace efficiently using batched einsum
    trace_hat = torch.einsum('bni,bin->b', phi, p_matrix)
    
    # RSS = ||Y - Y_hat||^2
    squared_error = torch.sum(
        (predictions.squeeze(-1) - y_data.squeeze(-1))**2, dim=1
    )
    
    degrees_of_freedom = n_samples - trace_hat
    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)
    
    return squared_error / degrees_of_freedom
#: </p_matrix>

#: <t_stats>
def _compute_t_statistics(
    p_matrix: torch.Tensor, 
    noise_variance: torch.Tensor, 
    adaptive_coeffs: torch.Tensor
) -> torch.Tensor:
    r"""Computes the t-statistics for each coefficient.

    Args:
        p_matrix (torch.Tensor): The batched solver matrix P.
        noise_variance (torch.Tensor): The estimated noise variance (sigma^2).
        adaptive_coeffs (torch.Tensor): The fitted model coefficients (beta_hat).

    Returns:
        torch.Tensor: The t-statistics for each coefficient in each group.
        Shape: (n_groups, n_coeffs)
    """
    # Calculate diag(P @ P.H) efficiently
    var_beta_hat_diag = torch.einsum('bij,bji->bi', p_matrix, p_matrix.mT)
    std_err_beta = torch.sqrt(noise_variance.unsqueeze(-1) * var_beta_hat_diag)
    
    # Avoid division by zero
    std_err_beta[std_err_beta == 0] = 1e-9 
    
    # FIX: Align the coefficients tensor device with the computed standard errors
    adaptive_coeffs = adaptive_coeffs.to(std_err_beta.device)
    
    return adaptive_coeffs.squeeze(-1) / std_err_beta
#: </t_stats>

def run_diagnostics(model: StaticTAM, data_train: pd.DataFrame) -> torch.Tensor:
    r"""Runs the complete statistical diagnostic pipeline for a trained model.

    This function:
    1.  Prepares the training data (balancing, tensorizing).
    2.  Builds the design matrix (Phi) and penalty matrix (P).
    3.  Computes the coefficient solver matrix (P).
    4.  Estimates the noise variance (sigma^2).
    5.  Computes the t-statistics for all coefficients.
    6.  Calls `display_global_significance` to report the results.

    Args:
        model (StaticTAM): A fitted `StaticTAM` model instance.
        data_train (pd.DataFrame): The DataFrame used to train the model.

    Returns:
        torch.Tensor: The computed t-statistics.
        Shape: (n_groups, n_total_coeffs)
    """
    if model.coefficients_ is None:
        raise RuntimeError("Model must be fitted before running diagnostics.")
        
    print("\n--- Running Full Model Diagnostics ---")
    
    # 1. Prepare Data (re-create the exact data used for fitting)
    _, balanced_data = _balance_groups(
        dataset=data_train, 
        group_col=model.group_col_, 
        date_col=model.date_col_,
        method="drop"
    )
    
    x_data, y_data, _ = model._prepare_data(balanced_data, target_col=model.target_col_)
    
    # 2. Build Matrices
    phi = model._build_design_matrix(x_data) 
    penalty_M_star_M = model._build_penalty_matrix()

    predictions = _predict_from_coeffs(phi, model.coefficients_)

    n_samples = x_data.shape[1]
    
    # 3. Compute Stats
    print("Computing P-matrix (solver)...")
    p_matrix = _compute_p_matrix(phi, penalty_M_star_M, n_samples)
    print("Estimating noise variance (sigma^2)...")
    noise_variance = _estimate_noise_variance(phi, p_matrix, y_data, predictions, n_samples)
    print("Computing t-statistics...")
    t_statistics = _compute_t_statistics(p_matrix, noise_variance, model.coefficients_)

    # 4. Display Results
    display_global_significance(t_statistics, model)

    return t_statistics

#: <global_significance>
def display_global_significance(
    t_statistics: torch.Tensor, 
    model: StaticTAM
):
    r"""Aggregates and visualizes feature significance across all groups.

    Since complex terms (splines, Fourier) have many coefficients, individual
    t-stats are hard to interpret. This function computes a "Global Significance Stat".
    
    This aggregates the statistical power of the entire effect.

    Args:
        t_statistics (torch.Tensor): The computed t-statistics.
        model (StaticTAM): The fitted model instance.
    """
    import matplotlib.pyplot as plt
    
    if not model.effects_list_:
        print("Error: Model has no effects list.")
        return

    # Use the v2.0 summary() method to retrieve standardized labels
    if hasattr(model, 'summary'):
        df_summary = model.summary()
        # Filter out Offset as it's typically not relevant for feature importance
        df_feats = df_summary[df_summary['Type'] != 'Offset']
        feature_labels = [f"{row['Type']}({row['Feature']})" for _, row in df_feats.iterrows()]
    else:
        print("Error: Model does not support v2.0 summary().")
        return

    scores_by_feature = {label: [] for label in feature_labels}
    
    # Iterate over each group (batch dimension)
    for h in range(t_statistics.shape[0]): 
        t_stats_h = t_statistics[h] 
        coeff_idx = 0
        feat_counter = 0
        
        for effect in model.effects_list_:
            term_len = effect.get_n_coeffs()
            feature_t_stats = t_stats_h[coeff_idx : coeff_idx + term_len]
            
            if not isinstance(effect, OffsetEffect):
                # GlobalStat = n_coeffs * L2-norm(t_stats)
                global_stat = term_len * torch.sqrt(torch.sum((feature_t_stats)**2))
                
                if feat_counter < len(feature_labels):
                    label = feature_labels[feat_counter]
                    scores_by_feature[label].append(global_stat.item())
                    feat_counter += 1
                
            coeff_idx += term_len

    print("\n--- 'Global Significance Stat' (n_coeffs * L2-norm(t-stats)) Across All Groups ---")
    header = f"{'Feature Term':<30} | {'Min':>7} | {'Q1':>7} | {'Median':>7} | {'Mean':>7} | {'Q3':>7} | {'Max':>7}"
    print(header)
    print("-" * len(header))

    valid_scores = {k: v for k, v in scores_by_feature.items() if v}

    for name, scores in valid_scores.items():
        stats = np.percentile(scores, [0, 25, 50, 75, 100])
        mean_score = np.mean(scores)
        print(f"{name:<30} | {stats[0]:7.2f} | {stats[1]:7.2f} | {stats[2]:7.2f} | {mean_score:7.2f} | {stats[3]:7.2f} | {stats[4]:7.2f}")

    plot_data = list(valid_scores.values())
    plot_labels = list(valid_scores.keys())
    
    if not plot_data:
        print("No diagnostic data available to plot.")
        return

    means = [np.mean(scores) for scores in plot_data]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    boxplot = ax.boxplot(
        plot_data,
        orientation='horizontal',
        tick_labels=plot_labels,
        patch_artist=True, 
        medianprops={'color': 'black', 'linewidth': 2},
    )
    
    for patch in boxplot['boxes']:
        patch.set_facecolor('skyblue')

    # Add mean markers
    y_pos = np.arange(1, len(plot_data) + 1)
    ax.scatter(means, y_pos, marker='D', s=40, color='red', zorder=3, label='Mean')

    ax.set_title('Distribution of Feature Significance Across All Groups', fontsize=16)
    ax.set_xlabel('Global Significance Stat (Higher = More Significant)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
#: </global_significance>

#: <bootstrap>
def compute_bootstrap_significance(
    model: StaticTAM, 
    data: pd.DataFrame, 
    n_boot: int = 100, 
    block_size: int = 48 * 7 
) -> pd.DataFrame:
    r"""
    Estimates significance via Block Bootstrap.
    
    Provides robustness to autocorrelation, replacing standard t-stats 
    for time series data by resampling residual blocks.
    
    Args:
        model (StaticTAM): Fitted StaticTAM model.
        data (pd.DataFrame): Training data.
        n_boot (int): Number of bootstrap iterations.
        block_size (int): Size of the temporal block to resample (e.g., 1 week).
        
    Returns:
        pd.DataFrame: Robust t-statistics for each coefficient.
    """
    print(f"--- Running Block Bootstrap ({n_boot} iterations) ---")
    
    preds = model.predict(data)
    residuals = preds[f'Estimated{model.target_col_}'] - data[model.target_col_]
    
    res_values = residuals.values
    n = len(res_values)
    boot_coeffs = []
    
    y_pred_fixed = preds[f'Estimated{model.target_col_}'].values
    
    x_data, _, _ = model._prepare_data(data, target_col=model.target_col_)
    phi = model._build_design_matrix(x_data)
    device = phi.device 

    # Ensure penalty matrix is on the correct device
    penalty = model._build_penalty_matrix().to(device)
    n_samples_group = x_data.shape[1]
    
    # Initialize dummy tensors on the target device
    dummy_y = torch.zeros_like(phi[...,0:1], device=device)
    dummy_loss = torch.eye(1, device=device).to(phi.dtype)
    
    cov_X, _ = _compute_weighted_covariances(phi, dummy_y, dummy_loss)
    
    reg_term = (n_samples_group * penalty).unsqueeze(0)
    jitter = 1e-12 * torch.eye(cov_X.shape[-1], device=device).to(cov_X.dtype)
    
    # Pre-inversion of the Hessian matrix
    try:
        H_inv = torch.linalg.inv(cov_X + reg_term + jitter)
    except (RuntimeError, NotImplementedError):
        # Hardware fallback if native backend fails on inverse
        cov_X_cpu = cov_X.cpu()
        reg_term_cpu = reg_term.cpu()
        jitter_cpu = jitter.cpu()
        H_inv = torch.linalg.inv(cov_X_cpu + reg_term_cpu + jitter_cpu).to(device)
    
    for b in range(n_boot):
        # Draw residual blocks with replacement (Circular bootstrap)
        indices_start = np.random.randint(0, n, size=n // block_size + 1)
        res_boot = []
        for start in indices_start:
            indices = np.arange(start, start + block_size) % n
            res_boot.append(res_values[indices])
            
        res_boot = np.concatenate(res_boot)[:n]
        y_boot = y_pred_fixed + res_boot
        
        # Explicitly place the bootstrapped target on the correct device
        y_boot_tensor = torch.tensor(y_boot, device=device, dtype=phi.dtype).view(1, -1, 1)
        cov_XY_boot = phi.mT @ y_boot_tensor
        
        beta_b = H_inv @ cov_XY_boot
        boot_coeffs.append(beta_b.squeeze().cpu().numpy())
        
    boot_coeffs = np.array(boot_coeffs)
    std_coeffs = np.std(boot_coeffs, axis=0)
    mean_coeffs = np.mean(boot_coeffs, axis=0)
    
    t_stats_robust = mean_coeffs / (std_coeffs + 1e-9)
    
    return t_stats_robust
#: </bootstrap>