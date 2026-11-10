# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A missing target leaves the KalmanTAM state where it is and does not make it more confident: the mean is unchanged, the
covariance only grows by the process noise. A block that holds missing rows gives the update of the observed rows alone.
"""

import torch

from tam.model.kalman import _kalman_block_loop_optimized

D = 3
P0 = 4.0
R_OBS = 0.5
Q_DIAG = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)


def _run(phi, y, block_size=1, with_variance=False):
    base = torch.zeros_like(y)
    return _kalman_block_loop_optimized(
        phi, y, base, block_size, P0, R_OBS, Q_DIAG, 1e-12, 1.0, 1.0, with_variance, 1)


def _data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(1, n, D, generator=g, dtype=torch.float64)
    y = torch.randn(1, n, 1, generator=g, dtype=torch.float64)
    return phi, y


def _reference(phi, y, block_size):
    """The exact block filter on the observed rows of each block; the process noise of every step is added."""
    n = phi.shape[1]
    theta = torch.zeros(1, D, 1, dtype=torch.float64)
    p = torch.eye(D, dtype=torch.float64).unsqueeze(0) * P0
    q = torch.diag(Q_DIAG).unsqueeze(0)
    for a in range(0, n, block_size):
        b = min(a + block_size, n)
        rows = [i for i in range(a, b) if not torch.isnan(y[0, i, 0])]
        if rows:
            x = phi[:, rows]
            s = x @ p @ x.transpose(-2, -1) + R_OBS * torch.eye(len(rows), dtype=torch.float64)
            gain = p @ x.transpose(-2, -1) @ torch.linalg.inv(s)
            theta = theta + gain @ (y[:, rows] - x @ theta)
            p = p - gain @ x @ p
        p = p + q * (b - a)
    return theta, p


def test_a_run_of_missing_targets_leaves_the_state_mean_where_it_was():
    phi, y = _data(10)
    y_gap = y.clone()
    y_gap[:, 3:8] = float("nan")
    _, _, theta_before, _ = _run(phi[:, :3], y[:, :3])
    _, _, theta_after, _ = _run(phi[:, :8], y_gap[:, :8])
    assert torch.allclose(theta_after, theta_before, atol=1e-12)


def test_the_covariance_after_a_gap_is_the_covariance_before_plus_the_process_noise():
    phi, y = _data(10)
    k = 5
    y_gap = y.clone()
    y_gap[:, 3:3 + k] = float("nan")
    _, _, _, var_gap = _run(phi, y_gap, with_variance=True)
    _, p_ref = _reference(phi[:, :3], y[:, :3], 1)
    # The a-priori variance of the row after the gap is x (P + k Q) x' + R, with P the covariance after the 3 observed rows.
    x = phi[:, 3 + k:4 + k]
    expected = x @ (p_ref + k * torch.diag(Q_DIAG)) @ x.transpose(-2, -1) + R_OBS
    assert torch.allclose(var_gap[:, 3 + k:4 + k], expected, atol=1e-10)


def test_the_first_observation_after_a_gap_moves_the_state_with_the_gain_of_a_filter_that_waited():
    phi, y = _data(12, seed=3)
    y_gap = y.clone()
    y_gap[:, 4:10] = float("nan")
    _, _, theta, _ = _run(phi, y_gap)
    theta_ref, _ = _reference(phi, y_gap, 1)
    assert torch.allclose(theta, theta_ref, atol=1e-10)


def test_a_block_with_missing_rows_equals_the_filter_on_the_observed_rows():
    phi, y = _data(12, seed=1)
    y_gap = y.clone()
    y_gap[:, [1, 2, 6, 7, 9]] = float("nan")
    _, _, theta, _ = _run(phi, y_gap, block_size=4)
    theta_ref, _ = _reference(phi, y_gap, 4)
    assert torch.allclose(theta, theta_ref, atol=1e-9)


def test_a_series_without_missing_targets_is_unchanged():
    phi, y = _data(10, seed=2)
    pred, hist, theta, _ = _run(phi, y, block_size=3)
    theta_ref, _ = _reference(phi, y, 3)
    assert torch.allclose(theta, theta_ref, atol=1e-10)
    assert pred.shape == (1, 10, 1) and hist.shape[0] == 4
