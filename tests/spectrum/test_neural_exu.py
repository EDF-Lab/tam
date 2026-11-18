# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for Exp-Centered Units (ExU) in NeuralEffect."""

import numpy as np
import pytest
import torch

from tam.model.spectrum._neural import NeuralEffect


def test_neural_exu_step_discontinuity():
    """Verifies that ExU fits sharp step discontinuities with significantly lower error than smooth Tanh."""
    torch.manual_seed(42)
    n = 300
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(-1)
    # Piecewise step function with two sharp discontinuities
    y = ((x > 0.23).to(x.dtype) - 1.5 * (x > -0.41).to(x.dtype)).squeeze(-1)

    effect_exu = NeuralEffect(
        feature_name="x",
        n_neurons=64,
        activation="exu",
        lambda_p=1e-4,
        additional_features=None,
        seed=42,
        n_hidden_layers=1,
        extrapolate="linear",
    )
    phi_exu = effect_exu.build_feature_map(x)
    assert phi_exu.shape == (n, 64)
    assert torch.isfinite(phi_exu).all()

    # Linear least squares
    theta_exu = torch.linalg.lstsq(phi_exu, y.unsqueeze(-1)).solution
    pred_exu = (phi_exu @ theta_exu).squeeze(-1)

    effect_tanh = NeuralEffect(
        feature_name="x",
        n_neurons=64,
        activation="tanh",
        lambda_p=1e-4,
        additional_features=None,
        seed=42,
        n_hidden_layers=1,
        extrapolate="linear",
    )
    phi_tanh = effect_tanh.build_feature_map(x)
    theta_tanh = torch.linalg.lstsq(phi_tanh, y.unsqueeze(-1)).solution
    pred_tanh = (phi_tanh @ theta_tanh).squeeze(-1)

    # Discontinuity regions: within 0.04 of jumps
    mask_jumps = ((torch.abs(x.squeeze(-1) - 0.23) < 0.04) |
                  (torch.abs(x.squeeze(-1) - (-0.41)) < 0.04))

    rmse_exu = torch.sqrt(torch.mean((pred_exu[mask_jumps] - y[mask_jumps]) ** 2)).item()
    rmse_tanh = torch.sqrt(torch.mean((pred_tanh[mask_jumps] - y[mask_jumps]) ** 2)).item()

    # Overall RMSE
    rmse_exu_tot = torch.sqrt(torch.mean((pred_exu - y) ** 2)).item()
    rmse_tanh_tot = torch.sqrt(torch.mean((pred_tanh - y) ** 2)).item()

    assert rmse_exu_tot < rmse_tanh_tot
    assert rmse_exu < rmse_tanh


def test_neural_exu_penalty_matrix():
    """Verifies that penalty matrix is isotropic positive definite."""
    effect = NeuralEffect(
        feature_name="x",
        n_neurons=32,
        activation="exu",
        lambda_p=2.0,
        additional_features=None,
        seed=123,
        n_hidden_layers=1,
        extrapolate="linear",
    )
    p = effect.build_penalty_matrix()
    assert p.shape == (32, 32)
    assert torch.allclose(p, torch.eye(32) * 2.0)


def test_syntax_reference_lists_exu(capsys):
    """The syntax reference names every activation that `n(...)` accepts."""
    from tam import StaticTAM

    StaticTAM.show_syntax()
    assert "'exu'" in capsys.readouterr().out
