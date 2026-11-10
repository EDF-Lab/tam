# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Unit tests for the memory-estimation helpers in ``tam.model._memory``.

These guard the anti-OOM heuristics that decide between the dense direct
solver and the matrix-free CG path, and that size data/window chunks.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import tam
from tam.common.hardware import hw
from tam.model import _dispatcher
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


# --- group chunking of the direct solver: the D x D covariance of a chunk is part of its budget ---

G_PANEL, N_PANEL, D_PANEL = 200, 20, 100
BUDGET_FRACTION = 0.8            # share of the free memory the dispatcher allows itself


def _spy_direct_solver(monkeypatch, available_bytes):
    """Run the chunked direct solver on a (G, N, 1) panel of D features with the covariance and solve stubbed out."""
    chunks = []

    def fake_phi(x, effects):
        return torch.zeros(x.shape[0], x.shape[1], D_PANEL, dtype=torch.float64)

    def fake_cov(phi, y, loss_matrix, weights=None):
        chunks.append(phi.shape[0])
        return torch.zeros(phi.shape[0], D_PANEL, D_PANEL, dtype=torch.float64), torch.zeros(phi.shape[0], D_PANEL, 1, dtype=torch.float64)

    def fake_solve(cov_x, cov_xy, penalty, n):
        return torch.arange(cov_x.shape[0], dtype=torch.float64).view(-1, 1, 1).expand(-1, D_PANEL, 1).clone()

    monkeypatch.setattr(hw, "get_available_memory", lambda: available_bytes)
    monkeypatch.setattr(_dispatcher, "build_phi_from_effects", fake_phi)
    monkeypatch.setattr(_dispatcher, "_compute_weighted_covariances", fake_cov)
    monkeypatch.setattr(_dispatcher, "solve_linear_system", fake_solve)
    x = torch.zeros(G_PANEL, N_PANEL, 1, dtype=torch.float64)
    y = torch.zeros(G_PANEL, N_PANEL, 1, dtype=torch.float64)
    coeffs = _dispatcher._run_chunked_direct_solver(
        x, y, [], torch.zeros(D_PANEL, D_PANEL, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), N_PANEL, D_PANEL)
    return chunks, coeffs


def test_direct_solver_group_chunk_keeps_its_covariance_inside_the_memory_budget(monkeypatch):
    available = 12e6
    chunks, coeffs = _spy_direct_solver(monkeypatch, available)
    covariance_bytes = D_PANEL * D_PANEL * 8 * 2
    assert max(chunks) * covariance_bytes <= BUDGET_FRACTION * available
    assert sum(chunks) == G_PANEL


def test_direct_solver_keeps_the_group_order_across_chunks(monkeypatch):
    chunks, coeffs = _spy_direct_solver(monkeypatch, 12e6)
    assert len(chunks) > 1
    expected = torch.cat([torch.arange(c, dtype=torch.float64) for c in chunks])
    assert torch.equal(coeffs[:, 0, 0], expected)


def test_direct_solver_takes_the_whole_panel_in_one_chunk_when_it_fits(monkeypatch):
    chunks, _ = _spy_direct_solver(monkeypatch, 64e9)
    assert chunks == [G_PANEL]


def test_a_panel_too_large_for_one_batch_stays_on_the_direct_solver_and_is_chunked(monkeypatch):
    # One group fits (so the exact solver is kept), the whole panel does not: the groups are chunked, not sent to the CG.
    calls = {"direct": 0, "cg": 0}
    monkeypatch.setattr(hw, "get_available_memory", lambda: 12e6)
    monkeypatch.setattr(_dispatcher, "build_phi_from_effects", lambda x, e: torch.zeros(x.shape[0], x.shape[1], D_PANEL, dtype=torch.float64))
    monkeypatch.setattr(_dispatcher, "_run_chunked_direct_solver", lambda *a, **k: calls.__setitem__("direct", calls["direct"] + 1) or "direct")
    monkeypatch.setattr(_dispatcher, "_run_sparse_cg_solver", lambda *a, **k: calls.__setitem__("cg", calls["cg"] + 1) or "cg")
    x = torch.zeros(G_PANEL, N_PANEL, 1, dtype=torch.float64)
    out = _dispatcher.smart_solve(x, x, [], torch.zeros(D_PANEL, D_PANEL, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), N_PANEL)
    assert out == "direct" and calls == {"direct": 1, "cg": 0}


def test_a_single_group_that_does_not_fit_goes_to_the_matrix_free_solver(monkeypatch):
    calls = {"cg": 0}
    monkeypatch.setattr(hw, "get_available_memory", lambda: 1e5)   # below the footprint of one D x D system
    monkeypatch.setattr(_dispatcher, "build_phi_from_effects", lambda x, e: torch.zeros(x.shape[0], x.shape[1], D_PANEL, dtype=torch.float64))
    monkeypatch.setattr(_dispatcher, "_run_chunked_direct_solver", lambda *a, **k: pytest.fail("direct solver chosen for a system that does not fit"))
    monkeypatch.setattr(_dispatcher, "_run_sparse_cg_solver", lambda *a, **k: calls.__setitem__("cg", 1) or "cg")
    x = torch.zeros(G_PANEL, N_PANEL, 1, dtype=torch.float64)
    assert _dispatcher.smart_solve(x, x, [], torch.zeros(D_PANEL, D_PANEL, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), N_PANEL) == "cg"


def test_a_panel_fit_gives_the_same_coefficients_whether_its_groups_are_chunked_or_not_to_rounding(monkeypatch):
    rng = np.random.default_rng(0)
    n_groups, n = 12, 60
    frame = pd.DataFrame({
        "timestamp": np.tile(pd.date_range("2023-01-01", periods=n, freq="D"), n_groups),
        "grp": np.repeat(np.arange(n_groups), n),
        "x": rng.uniform(0, 1, n_groups * n),
    })
    frame["y"] = np.sin(6 * frame["x"]) + 0.1 * frame["grp"] + rng.normal(0, 0.1, len(frame))

    def coefficients():
        model = tam.StaticTAM(formula="y ~ s(x, k=8)", group_col="grp", date_col="timestamp")
        model.fit(frame)
        return model.coefficients_.detach().cpu().numpy().copy()

    whole = coefficients()
    monkeypatch.setattr(hw, "get_available_memory", lambda: 6e4)   # two or three groups per chunk
    chunked = coefficients()
    assert whole.shape == chunked.shape
    # One chunk is the unchanged path; splitting the batch moves the coefficients by rounding only (about 1e-11 relative).
    assert np.allclose(whole, chunked, rtol=1e-8, atol=1e-9)
