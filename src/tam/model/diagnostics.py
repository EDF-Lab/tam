# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

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
    offset = model._offset_of(x_data)
    if offset is not None:
        predictions = predictions + offset.to(device=predictions.device, dtype=predictions.dtype)

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
    offset = model._offset_of(x_data)  # known part of the linear predictor: it is not projected on the columns of Phi
    if offset is not None:
        offset = offset.to(device=device, dtype=phi.dtype)

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
        if offset is not None:
            y_boot_tensor = y_boot_tensor - offset
        cov_XY_boot = phi.mT @ y_boot_tensor
        
        beta_b = H_inv @ cov_XY_boot
        boot_coeffs.append(beta_b.squeeze().cpu().numpy())
        
    boot_coeffs = np.array(boot_coeffs)
    std_coeffs = np.std(boot_coeffs, axis=0)
    mean_coeffs = np.mean(boot_coeffs, axis=0)
    
    t_stats_robust = mean_coeffs / (std_coeffs + 1e-9)
    
    return t_stats_robust
#: </bootstrap>


#: <bispectrum_diagnostics>
def hinich_bispectrum_test(
    x: Any,
    n_fft: int = 128,
    overlap: float = 0.5,
    window: str = "hann",
    alpha: float = 0.01,
    top_k: int = 8,
) -> Dict[str, Any]:
    r"""Performs Hinich higher-order spectral analysis and tests for Gaussianity and linearity.

    Estimates the Welch-segmented bispectrum \hat{B}(k_1, k_2) and normalized bicoherence
    b^2(k_1, k_2) in the principal non-redundant triangle 1 <= k_2 <= k_1, k_1 + k_2 <= n_fft // 2.
    Computes the Hinich asymptotic Chi-square test of Gaussianity and linearity dispersion metrics,
    and extracts the top coupled frequency triads driving non-linear quadratic phase coupling (QPC).

    Args:
        x (Union[np.ndarray, torch.Tensor, pd.Series, list]): Input 1D time series.
        n_fft (int): Segment window length for FFT decomposition (must be even).
        overlap (float): Fractional segment overlap in [0, 1). Default 0.5 (50% overlap).
        window (str): Taper window function ('hann' or 'boxcar').
        alpha (float): Significance level for Gaussianity test (e.g. 0.01).
        top_k (int): Number of top coupled frequency triads to extract.

    Returns:
        Dict[str, Any]: Diagnostic results containing:
            - 'p_value_gaussianity': Asymptotic p-value for Gaussianity null hypothesis.
            - 'stat_gaussianity': Hinich chi-square test statistic.
            - 'df_gaussianity': Degrees of freedom (2 * number of grid points).
            - 'is_gaussian': True if p_value > alpha.
            - 'linearity_iqr': Interquartile range of bicoherence across principal domain.
            - 'linearity_std': Standard deviation of bicoherence across principal domain.
            - 'top_triads': List of dictionaries for top-K coupled frequency triads.
            - 'bicoherence_map': 2D ndarray of bicoherence values of shape (n_fft//2 + 1, n_fft//2 + 1).
            - 'principal_mask': Boolean mask of points in the principal domain.
            - 'n_segments': Number of Welch segments processed.
            - 'n_fft': FFT window length.
    """
    from scipy.stats import chi2

    if isinstance(x, torch.Tensor):
        x_arr = x.detach().cpu().numpy().astype(np.float64).ravel()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        x_arr = x.values.astype(np.float64).ravel()
    else:
        x_arr = np.asarray(x, dtype=np.float64).ravel()

    n_samples = len(x_arr)
    n_fft = int(n_fft)
    if n_samples < n_fft:
        raise ValueError(
            f"Input series length ({n_samples}) must be at least n_fft ({n_fft}) for bispectrum analysis."
        )

    # Demean the signal
    x_zero_mean = x_arr - np.mean(x_arr)

    # Window function
    if window == "hann":
        w = np.hanning(n_fft)
    else:
        w = np.ones(n_fft, dtype=np.float64)

    # Overlapping segmentation
    step = max(1, int(n_fft * (1.0 - overlap)))
    n_segs = (n_samples - n_fft) // step + 1
    if n_segs < 1:
        raise ValueError(f"Insufficient samples ({n_samples}) to form any segment of length {n_fft}.")

    segments = np.lib.stride_tricks.sliding_window_view(x_zero_mean, n_fft)[::step][:n_segs]
    win_segs = segments * w

    # Real FFT across segments: shape (n_segs, n_fft // 2 + 1)
    fft_coeffs = np.fft.rfft(win_segs, axis=-1)
    n_freqs = n_fft // 2 + 1

    # Principal non-redundant triangular domain of the frequency pairs (k_2 not above k_1, and k_1 + k_2 within the Nyquist range)
    principal_mask = np.zeros((n_freqs, n_freqs), dtype=bool)
    for k1 in range(1, n_freqs):
        for k2 in range(1, k1 + 1):
            if k1 + k2 <= n_fft // 2:
                principal_mask[k1, k2] = True

    coords = np.argwhere(principal_mask)
    k1_idx = coords[:, 0]
    k2_idx = coords[:, 1]
    k3_idx = k1_idx + k2_idx

    # Slices across segments
    x1 = fft_coeffs[:, k1_idx]
    x2 = fft_coeffs[:, k2_idx]
    x3 = fft_coeffs[:, k3_idx]

    product_12 = x1 * x2  # shape (n_segs, n_points)
    bispec_segment = product_12 * np.conj(x3)

    # Welch averaged bispectrum: shape (n_points,)
    bispec_mean = np.mean(bispec_segment, axis=0)

    # Normalization (Kim & Powers / Haubrich bicoherence):
    # E[|X(k1)X(k2)|^2] * E[|X(k1+k2)|^2]
    denom_p12 = np.mean(np.abs(product_12) ** 2, axis=0)
    denom_p3 = np.mean(np.abs(x3) ** 2, axis=0)
    denom = denom_p12 * denom_p3 + 1e-12

    bicoherence_vals = np.clip(np.abs(bispec_mean) ** 2 / denom, 0.0, 1.0)

    # Populate 2D bicoherence map (symmetric across diagonal)
    bicoherence_map = np.zeros((n_freqs, n_freqs), dtype=np.float64)
    bicoherence_map[k1_idx, k2_idx] = bicoherence_vals
    bicoherence_map[k2_idx, k1_idx] = bicoherence_vals

    # Hinich Gaussianity Test
    # Under Gaussianity null, 2 * n_segs * b^2(k1, k2) is asymptotically central chi-square with 2 d.o.f.
    stat_gaussianity = float(np.sum(2.0 * n_segs * bicoherence_vals))
    df_gaussianity = int(2 * len(bicoherence_vals))
    p_value_gaussianity = float(chi2.sf(stat_gaussianity, df_gaussianity))
    is_gaussian = bool(p_value_gaussianity > alpha)

    # Linearity metrics (dispersion across principal domain)
    q75, q25 = np.percentile(bicoherence_vals, [75, 25])
    linearity_iqr = float(q75 - q25)
    linearity_std = float(np.std(bicoherence_vals))

    # Top-K coupled frequency triads
    sorted_order = np.argsort(bicoherence_vals)[::-1]
    top_indices = sorted_order[: min(top_k, len(sorted_order))]

    top_triads = []
    for idx in top_indices:
        k1_val = int(k1_idx[idx])
        k2_val = int(k2_idx[idx])
        k3_val = int(k3_idx[idx])
        b2_val = float(bicoherence_vals[idx])
        b_val = bispec_mean[idx]

        top_triads.append({
            "k1": k1_val,
            "k2": k2_val,
            "k3": k3_val,
            "omega1": float(2.0 * np.pi * k1_val / n_fft),
            "omega2": float(2.0 * np.pi * k2_val / n_fft),
            "omega3": float(2.0 * np.pi * k3_val / n_fft),
            "f1": float(k1_val / n_fft),
            "f2": float(k2_val / n_fft),
            "f3": float(k3_val / n_fft),
            "bicoherence": b2_val,
            "bispectrum_mag": float(np.abs(b_val)),
            "bispectrum_phase": float(np.angle(b_val)),
        })

    return {
        "p_value_gaussianity": p_value_gaussianity,
        "stat_gaussianity": stat_gaussianity,
        "df_gaussianity": df_gaussianity,
        "is_gaussian": is_gaussian,
        "linearity_iqr": linearity_iqr,
        "linearity_std": linearity_std,
        "top_triads": top_triads,
        "bicoherence_map": bicoherence_map,
        "principal_mask": principal_mask,
        "n_segments": n_segs,
        "n_fft": n_fft,
    }
#: </bispectrum_diagnostics>
