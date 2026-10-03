# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Memory-safe Generalized Cross Validation (GCV) solver.

This module handles the Multiple Smoothing Parameter (MSP) estimation.
It dynamically routes matrix inversions and chunking based on available VRAM,
optimizing a dedicated lambda_p regularization parameter for each individual effect
using a discrete coordinate descent algorithm.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

from tam.common.hardware import hw
from .spectrum import BaseEffect, build_phi_from_effects
from ._math import compute_gcv_score, solve_linear_system
from ._memory import get_safe_chunk_size

#: <smart_solve_gcv>
def smart_solve_gcv(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    effects_list: List[BaseEffect],
    loss_matrix: torch.Tensor,
    alpha_p_bounds: Tuple[float, float],
    number_of_steps: int = 10,
    alpha_p_list: Optional[List[float]] = None,
    gamma: float = 1.4,
    verbose: bool = False
) -> Tuple[torch.Tensor, np.ndarray, float]:
    r"""
    Memory-safe Generalized Cross Validation (GCV) solver.
    Dynamically routes matrix inversions and chunking based on available VRAM.

    On return, every effect in `effects_list` carries the smoothing weights that were selected
    for it, so rebuilding the penalty from the effects reproduces the fit. A search
    that raises leaves the formula's own weights in place.

    The search runs over one coordinate per smoothing parameter: one per effect, and one per margin
    for a tensor product (`effect.penalty_coordinates()`). The returned `best_lambda_ps` is the flat
    vector of all coordinates, effect after effect; `effect.n_penalty_coordinates` splits it.
    """
    run_device = x_data.device
    num_samples = x_data.shape[1]
    n_groups = x_data.shape[0]
    
    dummy_x = x_data[:, 0:1, :].to(run_device)
    dummy_phi = build_phi_from_effects(dummy_x, effects_list)
    total_d = dummy_phi.shape[-1]
    del dummy_x, dummy_phi
    
    if total_d > 7500:
        raise MemoryError(
            f"Feature dimension D={total_d} is too massive for Generalized Cross Validation (GCV). "
            "GCV requires computing the exact trace of the dense inverse covariance matrix, "
            "which will cause a severe Out-Of-Memory (OOM) crash on your GPU. "
            "Please use `grid_search_fit()` instead, which utilizes matrix-free Conjugate Gradient routing."
        )

    spans = []
    c_idx = 0
    for e in effects_list:
        k = e.get_n_coeffs()
        spans.append((c_idx, c_idx + k))
        c_idx += k

    # One GCV coordinate per smoothing parameter: one per effect, except a tensor product, which has one per margin.
    coords_per_effect = [e.n_penalty_coordinates for e in effects_list]
    coord_spans = []
    c_start = 0
    for count in coords_per_effect:
        coord_spans.append((c_start, c_start + count))
        c_start += count
    n_coords = c_start

    def sync_penalty_blocks(
        penalty: torch.Tensor,
        applied_alpha_ps: np.ndarray,
        alpha_ps: np.ndarray,
        device: torch.device
    ) -> None:
        r"""
        Writes the penalty blocks for `alpha_ps`, rebuilding only the ones that moved.

        Each block is produced by the effect itself at lambda_p = 10**alpha, so the
        matrix scored here is exactly the one `fit()` reassembles from the lambdas this
        search returns. Rescaling a prebuilt block would instead score the product of
        the formula's own lambda_p and the tested one, leaving the selected weight
        unable to reproduce the model it was selected on.

        The trial weight lives on the effect only while its block is built: a search
        that raises must not leave the caller's effects holding a candidate.
        """
        for i, effect in enumerate(effects_list):
            c0, c1 = coord_spans[i]
            if np.array_equal(alpha_ps[c0:c1], applied_alpha_ps[c0:c1]):
                continue

            formula_coordinates = effect.penalty_coordinates()
            try:
                effect.set_penalty_coordinates([float(10.0 ** a) for a in alpha_ps[c0:c1]])
                mat = effect.build_penalty_matrix()
            finally:
                effect.set_penalty_coordinates(formula_coordinates)

            if mat.is_sparse:
                mat = mat.to_dense()

            start, end = spans[i]
            penalty[start:end, start:end] = mat.to(device)
            applied_alpha_ps[c0:c1] = alpha_ps[c0:c1]

    bytes_per_group = (total_d * total_d * 8) * 5
    total_bytes = bytes_per_group * n_groups
    available_mem = hw.get_available_memory()
    
    def _get_chunked_covs(x_subset: torch.Tensor, y_subset: torch.Tensor):
        nonlocal run_device
        n_groups_in_subset = x_subset.shape[0]
        num_samples_in_subset = x_subset.shape[1]
        
        cov_x_total = torch.zeros((n_groups_in_subset, total_d, total_d), dtype=torch.get_default_dtype(), device=run_device)
        cov_xy_total = torch.zeros((n_groups_in_subset, total_d, y_subset.shape[-1]), dtype=torch.get_default_dtype(), device=run_device)
        Y_sq_total = torch.zeros(n_groups_in_subset, dtype=torch.get_default_dtype(), device=run_device)
        
        available_bytes = hw.get_available_memory()
        allocatable_bytes = available_bytes * 0.8
        bytes_per_group_full_n = num_samples_in_subset * total_d * 8 * 5.0 
        safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1
        
        g_start = 0
        while g_start < n_groups_in_subset:
            g_end = min(g_start + safe_group_batch, n_groups_in_subset)
            current_sub_batch_size = g_end - g_start
            
            try:
                x_chunk = x_subset[g_start:g_end, :, :].to(run_device)
                y_chunk = y_subset[g_start:g_end, :, :].to(run_device)
                phi_chunk = build_phi_from_effects(x_chunk, effects_list)
                
                if loss_matrix.shape[0] == 1:
                    L_sqrt = loss_matrix[0, 0].sqrt()
                    phi_weighted = phi_chunk * L_sqrt
                    cov_x_total[g_start:g_end] = phi_weighted.mT @ phi_weighted
                    cov_xy_total[g_start:g_end] = phi_chunk.mT @ (y_chunk * loss_matrix[0, 0])
                    Y_sq_total[g_start:g_end] = torch.sum(torch.abs(y_chunk)**2 * loss_matrix[0, 0], dim=1).squeeze(-1)
                    del phi_weighted, L_sqrt
                else:
                    cov_x_total[g_start:g_end] = phi_chunk.mT @ phi_chunk
                    y_weighted = y_chunk.to(phi_chunk.dtype) @ loss_matrix
                    cov_xy_total[g_start:g_end] = phi_chunk.mT @ y_weighted
                    Y_sq_total[g_start:g_end] = torch.sum(torch.abs(y_chunk)**2, dim=1).squeeze(-1)
                    del y_weighted
                    
                del x_chunk, y_chunk, phi_chunk
                g_start += current_sub_batch_size
                
            except (torch.OutOfMemoryError, MemoryError):
                if safe_group_batch > 1:
                    safe_group_batch, run_device = hw.handle_oom(
                        current_batch=safe_group_batch, 
                        device=run_device, 
                        context="GCV covariance group reduction", 
                        allow_cpu_fallback=False
                    )
                    continue
                else:
                    raise RuntimeError("A single full group exceeds available physical memory during GCV covariance computation.")
                
        return cov_x_total, cov_xy_total, Y_sq_total

    initial_alpha_ps = np.array([
        np.log10(c) if c > 0 else alpha_p_bounds[0]
        for e in effects_list for c in e.penalty_coordinates()
    ], dtype=np.float64)

    if alpha_p_list is None:
        # The local steps stay inside the box, so the seed has to start inside it.
        # An explicit coordinate list ignores the bounds, and so does its seed.
        initial_alpha_ps = np.clip(initial_alpha_ps, alpha_p_bounds[0], alpha_p_bounds[1])

    step_size = (alpha_p_bounds[1] - alpha_p_bounds[0]) / float(number_of_steps)

    if total_bytes < available_mem * 0.3:
        if verbose:
            print("[GCV Engine] VRAM footprint < 30%. Caching covariances globally on GPU.")
        cov_X, cov_XY, Y_sq = _get_chunked_covs(x_data, y_data)
        current_penalty = torch.zeros((total_d, total_d), dtype=torch.get_default_dtype(), device=run_device)
        applied_alpha_ps = np.full(n_coords, np.nan, dtype=np.float64)

        def gcv_objective(alpha_ps: np.ndarray) -> float:
            sync_penalty_blocks(current_penalty, applied_alpha_ps, alpha_ps, run_device)

            score = compute_gcv_score(
                cov_X=cov_X, 
                cov_XY=cov_XY, 
                Y_sq=Y_sq,
                penalty_M_star_M=current_penalty, 
                lambda_p=1.0, 
                n_samples=num_samples,
                gamma=gamma
            )
            return score.item()

        current_alpha_ps = np.copy(initial_alpha_ps)
        best_gcv = gcv_objective(current_alpha_ps)
        
        cycle = 0
        max_cycles = 15
        
        while cycle < max_cycles:
            cycle += 1
            improved_in_cycle = False
            
            for i in range(n_coords):
                original_val = current_alpha_ps[i]
                best_val_for_effect = original_val
                
                if alpha_p_list is not None:
                    candidates = alpha_p_list
                else:
                    candidates = [
                        np.clip(original_val - step_size, alpha_p_bounds[0], alpha_p_bounds[1]),
                        np.clip(original_val + step_size, alpha_p_bounds[0], alpha_p_bounds[1])
                    ]
                    
                for cand in candidates:
                    if np.isclose(cand, original_val):
                        continue
                        
                    current_alpha_ps[i] = cand
                    score = gcv_objective(current_alpha_ps)
                    
                    if score < best_gcv:
                        best_gcv = score
                        best_val_for_effect = cand
                        improved_in_cycle = True
                        
                current_alpha_ps[i] = best_val_for_effect
                
            if not improved_in_cycle:
                break
                
        best_lambda_ps = 10.0 ** current_alpha_ps

        sync_penalty_blocks(current_penalty, applied_alpha_ps, current_alpha_ps, run_device)
        for effect, (c0, c1) in zip(effects_list, coord_spans):
            effect.set_penalty_coordinates([float(v) for v in best_lambda_ps[c0:c1]])

        coeffs = solve_linear_system(cov_X, cov_XY, current_penalty, num_samples)
        
        return coeffs, best_lambda_ps, best_gcv

    else:
        eval_device = run_device if bytes_per_group < available_mem * 0.4 else torch.device('cpu')
        if eval_device.type == 'cpu':
            print("Notice: KxK matrix inversions exceed GPU VRAM. Routing solver to CPU.")
        else:
            print("Notice: Total system exceeds VRAM. Processing GCV in group chunks.")
            
        cov_X_list, cov_XY_list, Y_sq_list = [], [], []
        for g in range(n_groups):
            cx, cxy, y_sq = _get_chunked_covs(x_data[g:g+1], y_data[g:g+1])
            cov_X_list.append(cx.cpu())
            cov_XY_list.append(cxy.cpu())
            Y_sq_list.append(y_sq.cpu())
            hw.empty_cache()
            
        current_penalty = torch.zeros((total_d, total_d), dtype=torch.get_default_dtype(), device=eval_device)
        applied_alpha_ps = np.full(n_coords, np.nan, dtype=np.float64)

        def gcv_objective(alpha_ps: np.ndarray) -> float:
            try:
                sync_penalty_blocks(current_penalty, applied_alpha_ps, alpha_ps, eval_device)
            except (torch.OutOfMemoryError, MemoryError):
                # Basis assembly runs on the compute device even when scoring was
                # offloaded to the host; a block that does not fit rejects the
                # candidate instead of aborting the search.
                hw.empty_cache()
                return float('inf')

            total_score = 0.0
            for g in range(n_groups):
                try:
                    score = compute_gcv_score(
                        cov_X=cov_X_list[g].to(eval_device), 
                        cov_XY=cov_XY_list[g].to(eval_device), 
                        Y_sq=Y_sq_list[g].to(eval_device),
                        penalty_M_star_M=current_penalty, 
                        lambda_p=1.0, 
                        n_samples=num_samples,
                        gamma=gamma
                    )
                    total_score += score.item()
                except (torch.OutOfMemoryError, MemoryError):
                    hw.empty_cache()
                    return float('inf')
            return total_score / n_groups

        current_alpha_ps = np.copy(initial_alpha_ps)
        best_gcv = gcv_objective(current_alpha_ps)
        
        cycle = 0
        max_cycles = 15
        
        while cycle < max_cycles:
            cycle += 1
            improved_in_cycle = False
            
            for i in range(n_coords):
                original_val = current_alpha_ps[i]
                best_val_for_effect = original_val
                
                if alpha_p_list is not None:
                    candidates = alpha_p_list
                else:
                    candidates = [
                        np.clip(original_val - step_size, alpha_p_bounds[0], alpha_p_bounds[1]),
                        np.clip(original_val + step_size, alpha_p_bounds[0], alpha_p_bounds[1])
                    ]
                    
                for cand in candidates:
                    if np.isclose(cand, original_val):
                        continue
                        
                    current_alpha_ps[i] = cand
                    score = gcv_objective(current_alpha_ps)
                    
                    if score < best_gcv:
                        best_gcv = score
                        best_val_for_effect = cand
                        improved_in_cycle = True
                        
                current_alpha_ps[i] = best_val_for_effect
                
            if not improved_in_cycle:
                break
                
        best_lambda_ps = 10.0 ** current_alpha_ps

        sync_penalty_blocks(current_penalty, applied_alpha_ps, current_alpha_ps, eval_device)
        for effect, (c0, c1) in zip(effects_list, coord_spans):
            effect.set_penalty_coordinates([float(v) for v in best_lambda_ps[c0:c1]])

        group_coeffs = []
        for g in range(n_groups):
            coeffs_g = solve_linear_system(
                cov_X_list[g].to(eval_device), 
                cov_XY_list[g].to(eval_device), 
                current_penalty, 
                num_samples
            )
            group_coeffs.append(coeffs_g.cpu())
            
        coeffs = torch.cat(group_coeffs, dim=0)
        return coeffs, best_lambda_ps, best_gcv
#: </smart_solve_gcv>