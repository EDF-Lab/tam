# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._pid.PIDEffect``."""

import pytest
import torch

import tam
from tam.model.spectrum import PIDEffect


def test_pid_contract(normalized, penalty_shape):
    effect = PIDEffect("x", window=3, lambda_p=1.0, d_penalty_multiplier=10.0, extrapolate="continue")
    assert effect.get_n_coeffs() == 3  # proportional, integral, derivative

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 3)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (3, 3)


def test_pid_penalty_boosts_derivative_term():
    effect = PIDEffect("x", window=3, lambda_p=1.0, d_penalty_multiplier=10.0, extrapolate="continue")
    P = effect.build_penalty_matrix()
    diag = torch.diagonal(P)
    # Derivative (index 2) is penalized harder than P and I.
    assert diag[2] == 10.0 * diag[0]


def test_pid_zero_window_uses_full_cumulative_integral(normalized):
    """window=0 collapses the rolling-mean integral to the raw cumulative sum."""
    effect = PIDEffect("x", window=0, lambda_p=1.0, d_penalty_multiplier=1.0, extrapolate="continue")
    phi = effect.build_feature_map(normalized(2, 8))
    assert phi.shape == (2, 8, 3)
    assert torch.isfinite(phi).all()


def test_pid_negative_window_raises():
    with pytest.raises(ValueError, match="non-negative"):
        PIDEffect("x", window=-1, lambda_p=1.0, d_penalty_multiplier=1.0, extrapolate="continue")
