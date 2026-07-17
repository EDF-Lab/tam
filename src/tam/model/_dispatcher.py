# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Mathematical Solver Dispatcher for the TAM Framework.

This module acts as an intelligent routing layer between the statistical modeling 
abstractions and the low-level PyTorch linear algebra engines. 

The dispatcher dynamically evaluates the topological complexity (Feature Dimension D) 
and the empirical sample size (N) against the physical VRAM limits of the hardware. 
It seamlessly routes the system to either:
1.  An exact, chunked Direct Solver (for continuous, low-dimensional functional bases).
2.  A matrix-free Sparse Iterative Solver (Conjugate Gradient) to prevent Out-Of-Memory 
    (OOM) crashes when evaluating massive, high-dimensional algorithmic structures.

Reference:
    Wu, L., Yen, I., Chen, J., & Yan, R. (2018). Revisiting Random Binning Features: 
    Fast Convergence and Strong Parallelizability.
"""

import torch
import numpy as np
from typing import List, Dict

from tam.common.hardware import hw
from .spectrum import BaseEffect, build_phi_from_effects
from ._math import _compute_weighted_covariances, solve_linear_system, solve_sparse_cg, _predict_from_coeffs, _decompose_prediction_tensor
from ._memory import can_fit_dense_matrix, get_safe_chunk_size

#: <smart_solve_router>
def smart_solve(
    x_data: torch.Tensor, 
    y_data: torch.Tensor, 
    effects_list: List[BaseEffect], 
    penalty_matrix: torch.Tensor, 
    loss_matrix: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    r"""
    Dynamically routes the mathematical resolution to the optimal solver.
    """
    run_device = x_data.device
    dummy_x = x_data[:, 0:1, :].to(run_device)
    dummy_phi = build_phi_from_effects(dummy_x, effects_list)
    total_d = dummy_phi.shape[-1]
    del dummy_x, dummy_phi
    
    is_safe_for_direct_inversion = can_fit_dense_matrix(total_d, run_device, batch_size=1)
    
    if is_safe_for_direct_inversion:
        return _run_chunked_direct_solver(
            x_data, y_data, effects_list, penalty_matrix, loss_matrix, num_samples, total_d
        )       
    else:
        print(f"Notice: Feature dimension D={total_d} is massive.")
        print("Routing to matrix-free Conjugate Gradient (CG) solver to prevent VRAM exhaustion...")
        return _run_sparse_cg_solver(
            x_data, y_data, effects_list, penalty_matrix, loss_matrix, num_samples
            )
#: </smart_solve_router>

#: <chunked_direct_solver>
def _run_chunked_direct_solver(
    x_data: torch.Tensor, 
    y_data: torch.Tensor, 
    effects_list: List[BaseEffect], 
    penalty_matrix: torch.Tensor, 
    loss_matrix: torch.Tensor,
    num_samples: int,
    total_d: int
) -> torch.Tensor:
    r"""
    Executes an exact Direct Resolution using predictive Group Chunking 
    and reactive Sample Chunking.
    """
    run_device = x_data.device
    total_groups = x_data.shape[0]
    total_samples = x_data.shape[1]
    
    available_bytes = hw.get_available_memory()
    allocatable_bytes = available_bytes * 0.8
    bytes_per_group_full_n = total_samples * total_d * 8 * 5.0
    safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1

    all_coeffs = []
    g_start = 0

    while g_start < total_groups:
        g_end = min(g_start + safe_group_batch, total_groups)
        current_sub_batch_size = g_end - g_start

        try:
            x_sub = x_data[g_start:g_end].to(run_device)
            y_sub = y_data[g_start:g_end].to(run_device)
            phi_sub = build_phi_from_effects(x_sub, effects_list)
            cov_x_sub, cov_xy_sub = _compute_weighted_covariances(phi_sub, y_sub, loss_matrix)
            coeffs_sub = solve_linear_system(cov_x_sub, cov_xy_sub, penalty_matrix, num_samples)
            all_coeffs.append(coeffs_sub.cpu())
            
            del x_sub, y_sub, phi_sub, cov_x_sub, cov_xy_sub, coeffs_sub
            hw.empty_cache()
            
            g_start += current_sub_batch_size
            
        except (torch.OutOfMemoryError, MemoryError):
            if safe_group_batch > 1:
                safe_group_batch, run_device = hw.handle_oom(
                    current_batch=safe_group_batch, 
                    device=run_device, 
                    context="Group Reduction (Direct Solver)", 
                    allow_cpu_fallback=False
                )
            else:
                raise RuntimeError("A single full group (including all its samples) exceeds available physical memory. Please reduce the dataset size or switch to the CG solver.")
            continue

    return torch.cat(all_coeffs, dim=0)
#: </chunked_direct_solver>

#: <sparse_cg_solver>
def _run_sparse_cg_solver(
    x_data: torch.Tensor, 
    y_data: torch.Tensor, 
    effects_list: List[BaseEffect], 
    penalty_matrix: torch.Tensor, 
    loss_matrix: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    r"""
    Executes a matrix-free resolution using the Conjugate Gradient (CG) method.
    Uses strict Group-Chunking to guarantee Float64 deterministic scale-invariance.
    """
    run_device = x_data.device
    penalty_matrix = penalty_matrix.to(run_device)
    total_groups = x_data.shape[0]
    total_samples = x_data.shape[1]
    
    dummy_phi = build_phi_from_effects(x_data[0:1, 0:1, :].to(run_device), effects_list)
    total_d = dummy_phi.shape[-1]
    del dummy_phi
    
    available_bytes = hw.get_available_memory()
    allocatable_bytes = available_bytes * 0.8
    bytes_per_group_full_n = total_samples * total_d * 8 * 5.0
    safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1

    cov_xy_total = torch.zeros((total_groups, total_d, y_data.shape[-1]), dtype=torch.get_default_dtype(), device=run_device)
    g_start = 0
    
    while g_start < total_groups:
        g_end = min(g_start + safe_group_batch, total_groups)
        current_sub_batch_size = g_end - g_start
        try:
            x_chunk = x_data[g_start:g_end, :, :].to(run_device)
            y_chunk = y_data[g_start:g_end, :, :].to(run_device)
            
            phi_chunk = build_phi_from_effects(x_chunk, effects_list)
            y_data_aligned = y_chunk.to(phi_chunk.dtype)
            y_weighted = y_data_aligned @ loss_matrix
            
            cov_xy_total[g_start:g_end] = phi_chunk.mT @ y_weighted
            
            del x_chunk, y_chunk, phi_chunk, y_data_aligned, y_weighted
            g_start += current_sub_batch_size
            
        except (torch.OutOfMemoryError, MemoryError):
            if safe_group_batch > 1:
                safe_group_batch, run_device = hw.handle_oom(
                    current_batch=safe_group_batch, 
                    device=run_device, 
                    context="CG RHS Group Reduction", 
                    allow_cpu_fallback=False
                )
            else:
                raise RuntimeError("A single full group exceeds VRAM during CG RHS computation.")
            continue

    def compute_Av(v: torch.Tensor) -> torch.Tensor:
        nonlocal run_device, safe_group_batch
        v = v.to(run_device)
        At_v_total = torch.zeros_like(v)
        
        g_start = 0
        while g_start < total_groups:
            g_end = min(g_start + safe_group_batch, total_groups)
            current_sub_batch_size = g_end - g_start
            
            try:
                x_chunk = x_data[g_start:g_end, :, :].to(run_device)
                v_chunk = v[g_start:g_end, :, :]
                phi_chunk = build_phi_from_effects(x_chunk, effects_list)

                if loss_matrix.shape[0] == 1:
                    L_sqrt = loss_matrix[0, 0].sqrt()
                    phi_weighted = phi_chunk * L_sqrt
                    At_v_total[g_start:g_end] = phi_weighted.mT @ (phi_weighted @ v_chunk)
                    del phi_weighted, L_sqrt
                else:
                    At_v_total[g_start:g_end] = phi_chunk.mT @ (phi_chunk @ v_chunk)
                
                del x_chunk, phi_chunk, v_chunk
                g_start += current_sub_batch_size
                
            except (torch.OutOfMemoryError, MemoryError):
                if safe_group_batch > 1:
                    safe_group_batch, run_device = hw.handle_oom(
                        current_batch=safe_group_batch, 
                        device=run_device, 
                        context="CG Av Group Reduction", 
                        allow_cpu_fallback=False
                    )
                else:
                    raise RuntimeError("A single full group exceeds VRAM during CG Av computation.")
                continue
            
        if penalty_matrix.is_sparse:
            b_size, d_size, out_size = v.shape
            v_2d = v.transpose(0, 1).reshape(d_size, b_size * out_size)
            pv_2d = penalty_matrix @ v_2d
            pv_batched = pv_2d.reshape(d_size, b_size, out_size).transpose(0, 1)
            res = At_v_total + (num_samples * pv_batched)
        else:
            res = At_v_total + (num_samples * (penalty_matrix @ v))
            
        jitter_scale = 1e-6 * num_samples
        return res + (jitter_scale * v)

    cg_coeffs = solve_sparse_cg(
        compute_Av=compute_Av,
        b=cov_xy_total,
        tol=1e-6,
        max_iter=5000
    )
    
    hw.empty_cache()
    
    return cg_coeffs
#: </sparse_cg_solver>

#: <smart_evaluate>
def smart_evaluate_rmse(
    x_val: torch.Tensor, 
    y_val: torch.Tensor, 
    beta: torch.Tensor, 
    effects_list: List[BaseEffect]
) -> torch.Tensor:
    r"""
    Memory-safe evaluation of RMSE using group-level chunking with OOM auto-correction.
    """
    total_groups = x_val.shape[0]
    num_samples_val = x_val.shape[1]
    run_device = x_val.device
    
    dummy_phi = build_phi_from_effects(x_val[0:1, 0:1, :].to(run_device), effects_list)
    total_d = dummy_phi.shape[-1]
    del dummy_phi
    
    available_bytes = hw.get_available_memory()
    allocatable_bytes = available_bytes * 0.8
    # Memory for Phi, Beta, Preds, and Errors
    bytes_per_group_full_n = num_samples_val * total_d * 8 * 4.0 
    safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1
    
    sse = 0.0
    g_start = 0
    beta_eval = beta.to(run_device)
    
    while g_start < total_groups:
        g_end = min(g_start + safe_group_batch, total_groups)
        current_sub_batch_size = g_end - g_start
        
        try:
            x_chunk = x_val[g_start:g_end, :, :].to(run_device)
            y_chunk = y_val[g_start:g_end, :, :].to(run_device)
            
            phi_chunk = build_phi_from_effects(x_chunk, effects_list)
            beta_chunk = beta_eval[g_start:g_end]
            
            preds_chunk = _predict_from_coeffs(phi_chunk, beta_chunk)
            
            errors = y_chunk - preds_chunk
            sse += torch.sum(torch.square(errors)).item()
            
            del x_chunk, y_chunk, phi_chunk, preds_chunk, errors, beta_chunk
            g_start += current_sub_batch_size
            
        except (torch.OutOfMemoryError, MemoryError):
            if safe_group_batch > 1:
                safe_group_batch, run_device = hw.handle_oom(
                    current_batch=safe_group_batch, 
                    device=run_device, 
                    context="RMSE Evaluation Group Reduction", 
                    allow_cpu_fallback=True
                )
                beta_eval = beta.to(run_device)
                continue
            else:
                raise RuntimeError("A single full group exceeds VRAM during RMSE evaluation.")
                
    hw.empty_cache()
    total_elements = total_groups * num_samples_val
    return torch.tensor(np.sqrt(sse / total_elements), device=x_val.device)
#: </smart_evaluate>

#: <smart_decompose>
def smart_decompose(
    x_data: torch.Tensor, 
    beta: torch.Tensor, 
    effects_list: List[BaseEffect]
) -> Dict[str, torch.Tensor]:
    r"""
    Memory-safe decomposition of predictions using group-level chunking and OOM auto-correction.
    """
    total_groups = x_data.shape[0]
    num_samples = x_data.shape[1]
    run_device = x_data.device
    
    dummy_phi = build_phi_from_effects(x_data[0:1, 0:1, :].to(run_device), effects_list)
    total_d = dummy_phi.shape[-1]
    del dummy_phi
    
    available_bytes = hw.get_available_memory()
    allocatable_bytes = available_bytes * 0.8
    bytes_per_group_full_n = num_samples * total_d * 8 * 4.0 
    safe_group_batch = max(1, int(allocatable_bytes // bytes_per_group_full_n)) if bytes_per_group_full_n > 0 else 1
    
    accumulated_effects = {}
    g_start = 0
    beta_eval = beta.to(run_device)

    while g_start < total_groups:
        g_end = min(g_start + safe_group_batch, total_groups)
        current_sub_batch_size = g_end - g_start
        
        try:
            x_chunk = x_data[g_start:g_end, :, :].to(run_device)
            phi_chunk = build_phi_from_effects(x_chunk, effects_list)
            beta_chunk = beta_eval[g_start:g_end]
            
            chunk_effects = _decompose_prediction_tensor(
                phi_chunk, beta_chunk, effects_list
            )
            
            for k, v in chunk_effects.items():
                if k not in accumulated_effects:
                    accumulated_effects[k] = []
                accumulated_effects[k].append(v.cpu())
                
            del x_chunk, phi_chunk, chunk_effects, beta_chunk
            g_start += current_sub_batch_size
            
        except (torch.OutOfMemoryError, MemoryError):
            if safe_group_batch > 1:
                safe_group_batch, run_device = hw.handle_oom(
                    current_batch=safe_group_batch, 
                    device=run_device, 
                    context="decomposition group reduction", 
                    allow_cpu_fallback=True
                )
                beta_eval = beta.to(run_device)
                continue
            else:
                raise RuntimeError("A single full group exceeds available physical memory during decomposition.")
                
    hw.empty_cache()
    
    final_decomposed_effects = {
        k: torch.cat(v_list, dim=0) for k, v_list in accumulated_effects.items()
    }
    
    return final_decomposed_effects
#: </smart_decompose>