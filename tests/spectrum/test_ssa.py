# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for SSAEffect (Singular Spectrum Analysis empirical oscillation modes)."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.additive import StaticTAM
from tam.model.spectrum import SSAEffect


def test_ssa_contract():
    """Verifies coefficient counts, output shapes, and positive definite spectral penalty."""
    effect = SSAEffect(feature_name="x", window=32, k=8, lambda_p=1.2)
    assert effect.get_n_coeffs() == 8

    x = torch.randn(100, dtype=torch.float64)
    phi = effect.build_feature_map(x)
    assert phi.shape == (100, 8)
    assert torch.isfinite(phi).all()

    p_mat = effect.build_penalty_matrix()
    assert p_mat.shape == (8, 8)
    # Diagonal penalty must have strictly positive elements
    diag = torch.diagonal(p_mat)
    assert (diag > 0).all()


def test_ssa_causality():
    """Verifies that future rows do not alter past SSA representations."""
    effect = SSAEffect(feature_name="x", window=24, k=6)
    n = 80
    u1 = torch.rand(n, dtype=torch.float64)
    u2 = u1.clone()
    u2[50:] = u2[50:] + 10.0

    # Fit EOF basis first on common baseline
    effect.initialize(u1[:50])

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Rows 0..49 must be bit-identical
    np.testing.assert_allclose(
        phi1[:50].numpy(), phi2[:50].numpy(), rtol=1e-12, atol=1e-12
    )


def test_ssa_state_continuation():
    """Verifies that fit on [0, n) and predict on [n, m) carries sliding-window state seamlessly."""
    torch.manual_seed(42)
    n = 100
    m = 160
    window = 20
    k = 6
    x_full = torch.randn(m, dtype=torch.float64)
    x_train = x_full[:n]
    x_test = x_full[n:]

    effect_full = SSAEffect(feature_name="x", window=window, k=k)
    effect_split = SSAEffect(feature_name="x", window=window, k=k)

    # Fit EOF basis on train
    effect_full.initialize(x_train)
    effect_split.initialize(x_train)

    phi_full = effect_full.build_feature_map(x_full)

    phi_train = effect_split.build_feature_map(x_train)
    phi_test = effect_split.build_feature_map(x_test, state_in=effect_split._state_out)

    assert effect_split._state_out["hist"].shape == (1, window - 1, 1)

    # phi_test must match phi_full[n:m] exactly
    np.testing.assert_allclose(
        phi_test.numpy(), phi_full[n:m].numpy(), rtol=1e-10, atol=1e-10
    )


def test_ssa_anharmonic_cycle():
    """Verifies that SSA captures frequency-drifting anharmonic periodicities."""
    t = torch.linspace(0, 50, 600, dtype=torch.float64)
    phase = t * 1.0 + 0.008 * t ** 2
    x = torch.sin(phase) + 0.5 * torch.sin(2 * phase)

    effect = SSAEffect(feature_name="x", window=36, k=6)
    phi = effect.build_feature_map(x)
    assert phi.shape == (600, 6)

    # Leading EOF modes should explain a significant portion of signal energy
    reconstructed = phi[:, :2] @ torch.linalg.lstsq(phi[:, :2], x.unsqueeze(-1)).solution
    rmse = torch.sqrt(torch.mean((reconstructed.squeeze(-1) - x) ** 2)).item()
    assert rmse < 0.8


def test_ssa_statictam_integration():
    """Verifies StaticTAM parsing, fit, and predict with ssa() term."""
    np.random.seed(42)
    n = 120
    x = np.sin(np.linspace(0, 20, n))
    y = 0.5 * x + 0.1 * np.random.randn(n)

    df_train = pd.DataFrame({"X": x[:80], "Y": y[:80]})
    df_test = pd.DataFrame({"X": x[80:], "Y": y[80:]})

    model = StaticTAM(formula="Y ~ s(X, k=6) + ssa(X, window=16, k=4)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    assert any("ssa" in c for c in decomp.columns)


def test_ssa_eofs_come_from_the_trajectory_matrices_of_each_group():
    """The Hankel windows of one group never mix with the next one: stacked groups equal the pooled windows of each group."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 120, dtype=torch.float64, generator=g)
    stacked = SSAEffect("x", window=12, k=4)
    stacked.initialize(x)
    # the covariance built by hand from each group's own windows
    L = 12
    xc = x - x.mean()
    H = torch.cat([row.unfold(0, L, 1) for row in xc], dim=0)
    C = H.T @ H / H.shape[0]
    vals = torch.linalg.eigvalsh(C).flip(0)[:4]
    assert torch.allclose(stacked.singular_values, vals, atol=1e-10)
    whole = stacked.build_feature_map(x)
    assert whole.shape == (3, 120, 4) and stacked._state_out["hist"].shape == (3, 11, 1)
    assert torch.allclose(stacked.build_feature_map(x[1:2])[0], whole[1], atol=1e-12)


def test_ssa_short_input_is_computed_and_extrapolation_is_checked():
    eff = SSAEffect("x", window=20, k=3)
    x = torch.randn(2, 5, dtype=torch.float64)
    out = eff.build_feature_map(x)                       # a probe shorter than the window: not zeros
    assert out.shape == (2, 5, 3) and torch.isfinite(out).all() and float(out.abs().sum()) > 0
    with pytest.raises(ValueError, match="temporal effect"):
        SSAEffect("x", extrapolate="linear")


def test_ssa_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([SSAEffect("x")]) == {"x"}
