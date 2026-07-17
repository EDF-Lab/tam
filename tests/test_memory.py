# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for the memory-estimation helpers in ``tam.model._memory``.

These guard the anti-OOM heuristics that decide between the dense direct
solver and the matrix-free CG path, and that size data/window chunks.
"""

import tam
from tam.common.hardware import hw
from tam.model._memory import (
    can_fit_dense_matrix,
    get_safe_chunk_size,
    get_safe_window_batch_size,
)


def test_can_fit_dense_matrix_small_is_true():
    assert can_fit_dense_matrix(total_d=50, device=hw.device, batch_size=1) is True


def test_can_fit_dense_matrix_rejects_above_hard_limit():
    # Beyond max_safe_d the direct solver is refused regardless of free memory.
    assert can_fit_dense_matrix(total_d=10_000, device=hw.device, max_safe_d=7500) is False


def test_can_fit_dense_matrix_honors_custom_limit():
    assert can_fit_dense_matrix(total_d=100, device=hw.device, max_safe_d=50) is False


def test_get_safe_chunk_size_within_bounds():
    chunk = get_safe_chunk_size(total_samples=1000, total_d=500, device=hw.device)
    assert isinstance(chunk, int)
    assert 1 <= chunk <= 1000


def test_get_safe_chunk_size_never_exceeds_total_samples():
    # Tiny problem: the chunk is capped by the total number of samples.
    chunk = get_safe_chunk_size(total_samples=10, total_d=2, device=hw.device)
    assert chunk == 10


def test_get_safe_window_batch_size_is_positive():
    batch = get_safe_window_batch_size(num_samples_per_window=50, total_d=100, device=hw.device)
    assert isinstance(batch, int)
    assert batch >= 1
