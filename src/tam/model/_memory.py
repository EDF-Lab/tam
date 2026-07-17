# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Hardware Memory Management and Estimation.

This module isolates all low-level hardware interactions for the TAM 
framework. It calculates available VRAM/RAM, estimates the byte footprint of 
massive matrices, and computes safe algorithmic chunk sizes.

By treating the hardware as an independent oracle, this module guarantees that 
both continuous mathematical solvers and sparse iterative solvers strictly 
respect the physical limitations of the host machine, preventing PyTorch OOM crashes.
"""

import torch
from tam.common.hardware import hw

def can_fit_dense_matrix(
    total_d: int, 
    device: torch.device, 
    batch_size: int = 1,
    dtype_size: int = 8,  #  8 bytes for float64
    safety_factor: float = 4.0,
    max_safe_d: int = 7500
) -> bool:
    r"""
    Evaluates if a dense Covariance Matrix inversion can safely execute in VRAM.

    Inverting a matrix via LU or Cholesky decomposition requires allocating the 
    base matrix, the target vectors, and substantial temporary workspace memory 
    for the linear algebra backend (LAPACK for CPU, cuSOLVER/MAGMA for GPU).

    Args:
        total_d (int): The total feature dimension (D) of the Primal space.
        device (torch.device): The compute device.
        batch_size (int): The number of independent systems being solved simultaneously.
        dtype_size (int): Bytes per element (8 for float64).
        safety_factor (float): Multiplier accounting for backend workspace overhead.
        max_safe_d (int): Hard mathematical limit for numerical stability.

    Returns:
        bool: True if the exact direct solver is safe to use; False otherwise.
    """
    #  Hard threshold for numerical stability and acceptable direct-inversion compute time
    if total_d > max_safe_d:
        return False
        
    #  Compute the exact byte footprint of a (D x D) dense matrix
    matrix_bytes = batch_size * total_d * total_d * dtype_size
    
    #  Estimate total memory required for the solver operation
    required_bytes = matrix_bytes * safety_factor
    
    # Use HardwareManager instead of the deleted local function
    available_bytes = hw.get_available_memory()
    
    # Require that the operation takes no more than 90% of the currently free memory
    return required_bytes < (available_bytes * 0.9)

def get_safe_chunk_size(
    total_samples: int, 
    total_d: int, 
    device: torch.device, 
    dtype_size: int = 8,
    batch_size: int = 1
) -> int:
    r"""
    Calculates the maximum number of samples (N) that can be projected into 
    the Primal space simultaneously without exceeding memory limits.

    When constructing the Covariance matrix, the framework chunks 
    the dataset. The bottleneck is holding the intermediate design matrix 
    Phi_chunk of shape (Batch, N_chunk, D) in memory.

    Args:
        total_samples (int): The total number of empirical observations.
        total_d (int): The total feature dimension.
        device (torch.device): The compute device.
        dtype_size (int): Bytes per element (8 for float64).
        batch_size (int): Size of batch

    Returns:
        int: The safe batch size (number of rows) for incremental processing.
    """
    available_bytes = hw.get_available_memory()
    
    # Use HardwareManager backend knowledge to determine allocation caps
    if hw.backend != 'cpu':
        allocatable_bytes = available_bytes * 0.9
    else:
        allocatable_bytes = available_bytes * 0.7
    
    # Calculate bytes required per row.
    # A single row in Phi takes (1 * D * dtype_size). 
    # The matrix multiplication (Phi^H @ Phi) requires roughly 3x that for intermediate gradients.
    bytes_per_row = batch_size * total_d * dtype_size * 3.0
    
    # Prevent division by zero if D is somehow 0
    if bytes_per_row == 0:
        return total_samples
        
    safe_batch = int(allocatable_bytes // bytes_per_row)
    
    # Clamp the batch size between 1 and the total number of samples
    safe_batch = max(1, min(safe_batch, total_samples))
    
    return safe_batch

def get_safe_window_batch_size(
    num_samples_per_window: int, 
    total_d: int, 
    device: torch.device, 
    dtype_size: int = 8
) -> int:
    r"""
    Calculates the maximum number of sliding windows that can be solved 
    simultaneously in AdaptiveTAM without exceeding VRAM.
    """
    available_bytes = hw.get_available_memory()

    if hw.backend != 'cpu':
        allocatable_bytes = available_bytes * 0.8 # Use 80% of free VRAM
    else:
        allocatable_bytes = available_bytes * 0.6 # Use 60% of free RAM
    
    # Bytes required for a SINGLE window:
    #  Design Matrix (Phi): N_samples * D * 8 bytes
    size_phi = num_samples_per_window * total_d * dtype_size
    
    #  Covariance Matrix (Cov_X): D * D * 8 bytes
    size_cov = total_d * total_d * dtype_size
    
    # Multiply by ~5.0 to account for intermediate PyTorch gradients and the linear algebra backend workspaces
    bytes_per_window = (size_phi + 2 * size_cov) * 5.0
    
    if bytes_per_window == 0:
        return 1
        
    safe_batch = int(allocatable_bytes // bytes_per_window)
    return max(1, safe_batch)