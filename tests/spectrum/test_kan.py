# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for KANEffect (Kolmogorov-Arnold Network edge splines)."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.additive import StaticTAM
from tam.model.spectrum import KANEffect


def test_kan_contract():
    """Verifies coefficient counts, output shapes, and block-diagonal penalty matrix."""
    n_nodes = 6
    k_in = 5
    k_out = 7
    degree = 3
    expected_coeffs = n_nodes * k_out  # 6 * 7 = 42

    effect = KANEffect(
        feature_name="x1",
        additional_features=["x2", "x3"],
        n_nodes=n_nodes,
        k_inner=k_in,
        k_outer=k_out,
        spline_degree=degree,
        lambda_p=1.5,
    )

    assert effect.get_n_coeffs() == expected_coeffs

    x = torch.randn(50, 3, dtype=torch.float64)
    phi = effect.build_feature_map(x)
    assert phi.shape == (50, expected_coeffs)
    assert torch.isfinite(phi).all()

    p_mat = effect.build_penalty_matrix()
    assert p_mat.shape == (expected_coeffs, expected_coeffs)
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals >= -1e-10).all()

    # Verify per-feature standardization & node scale stability on disparately scaled features
    x_unscaled = torch.column_stack([
        x[:, 0] * 100.0 + 50.0,
        x[:, 1] * 0.01 - 2.0,
        x[:, 2] * 25.0 + 10.0,
    ])
    effect_unscaled = KANEffect(
        feature_name="x1",
        additional_features=["x2", "x3"],
        n_nodes=n_nodes,
        k_inner=k_in,
        k_outer=k_out,
        spline_degree=degree,
        lambda_p=1.5,
    )
    phi_unscaled = effect_unscaled.build_feature_map(x_unscaled)
    np.testing.assert_allclose(phi.numpy(), phi_unscaled.numpy(), rtol=1e-4, atol=1e-4)


def test_kan_multivariate_interaction_scaling():
    """Verifies KAN representation beats additive splines on multiplicative interaction x1 * x2."""
    torch.manual_seed(42)
    n = 500
    x1 = torch.rand(n, 1, dtype=torch.float64) * 2.0 - 1.0
    x2 = torch.rand(n, 1, dtype=torch.float64) * 2.0 - 1.0
    x = torch.cat([x1, x2], dim=-1)
    # Multiplicative interaction: f(x1, x2) = x1 * x2
    y = (x1 * x2).squeeze(-1)

    n_train = 350
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    effect = KANEffect(
        feature_name="x1",
        additional_features=["x2"],
        n_nodes=6,
        k_inner=4,
        k_outer=6,
        seed=42,
    )
    assert effect.get_n_coeffs() < 150

    phi_train = effect.build_feature_map(x_train)
    phi_test = effect.build_feature_map(x_test)

    # Solve linear outer weights with Ridge regularization
    reg = 1e-4 * torch.eye(phi_train.shape[1], dtype=torch.float64)
    H = phi_train.T @ phi_train + reg
    rhs = phi_train.T @ y_train
    theta = torch.linalg.solve(H, rhs)

    pred_test = phi_test @ theta
    rmse_kan = torch.sqrt(torch.mean((pred_test - y_test) ** 2)).item()

    # Additive spline baseline: s(x1) + s(x2)
    from tam.model.spectrum._kan import _cox_de_boor
    knots = torch.linspace(-1.2, 1.2, 10, dtype=torch.float64)
    phi_add_train = torch.cat([_cox_de_boor(x_train[:, 0], knots, 3), _cox_de_boor(x_train[:, 1], knots, 3)], dim=-1)
    phi_add_test = torch.cat([_cox_de_boor(x_test[:, 0], knots, 3), _cox_de_boor(x_test[:, 1], knots, 3)], dim=-1)

    theta_add = torch.linalg.lstsq(phi_add_train, y_train.unsqueeze(-1)).solution.squeeze(-1)
    pred_add = phi_add_test @ theta_add
    rmse_add = torch.sqrt(torch.mean((pred_add - y_test) ** 2)).item()

    # KAN should achieve substantially lower RMSE (> 2x) than strictly additive representation
    assert rmse_kan < rmse_add * 0.5


def test_kan_statictam_integration():
    """Verifies StaticTAM parsing, fit, and predict with kan() term."""
    np.random.seed(42)
    n = 100
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.cos(x1 * np.pi)
    x3 = np.sin(x1 * np.pi)
    y = x1 + 0.5 * (x2 * x3) + 0.1 * np.random.randn(n)

    df_train = pd.DataFrame({"X1": x1[:70], "X2": x2[:70], "X3": x3[:70], "Y": y[:70]})
    df_test = pd.DataFrame({"X1": x1[70:], "X2": x2[70:], "X3": x3[70:], "Y": y[70:]})

    model = StaticTAM(formula="Y ~ s(X1, k=6) + kan(X1, others='X2|X3', n_nodes=6, k_inner=4, k_outer=6)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    assert any("kan" in c for c in decomp.columns)


def test_kan_univariate_batch_layout():
    """One column read as (B, T) gives (B, T, K); the other layouts keep their shape."""
    effect = KANEffect("x", n_nodes=4, k_outer=6)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 500, dtype=torch.float64, generator=g)
    effect.initialize(x.unsqueeze(-1))
    phi = effect.build_feature_map(x)
    assert phi.shape == (2, 500, effect.get_n_coeffs())
    assert torch.allclose(phi, effect.build_feature_map(x.unsqueeze(-1)))
    assert effect.build_feature_map(x[0]).shape == (500, effect.get_n_coeffs())          # one series
    assert effect.build_feature_map(x[0].unsqueeze(-1)).shape == (500, effect.get_n_coeffs())   # (N, 1) rows of a tensor-product margin
    multi = KANEffect("x1", additional_features=["x2"], n_nodes=4, k_outer=6)
    assert multi.build_feature_map(torch.randn(30, 2, dtype=torch.float64)).shape == (30, multi.get_n_coeffs())
    assert multi.build_feature_map(torch.randn(3, 30, 2, dtype=torch.float64)).shape == (3, 30, multi.get_n_coeffs())


def test_kan_statistics_are_frozen_and_groups_independent():
    """Everything data-dependent is set at initialize; a group's features do not depend on the other groups."""
    effect = KANEffect("x1", additional_features=["x2"], n_nodes=4, k_outer=6)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 80, 2, dtype=torch.float64, generator=g)
    effect.initialize(x)
    mean, u_mean, knots = effect._x_mean.clone(), effect._u_mean.clone(), effect.knots_in.clone()
    whole = effect.build_feature_map(x)
    effect.build_feature_map(x * 5.0 + 2.0)
    assert torch.equal(effect._x_mean, mean) and torch.equal(effect._u_mean, u_mean) and torch.equal(effect.knots_in, knots)
    for b in range(3):
        assert torch.allclose(effect.build_feature_map(x[b:b + 1])[0], whole[b], atol=1e-12)
    assert effect.stateful is False


def test_kan_inner_parameters_do_not_depend_on_the_dtype():
    """The inner parameters are drawn in float64 on the CPU: float32 data gets the same values, cast."""
    a = KANEffect("x", n_nodes=4)
    b = KANEffect("x", n_nodes=4)
    a._init_inner_parameters(1, torch.float64, torch.device("cpu"))
    b._init_inner_parameters(1, torch.float32, torch.device("cpu"))
    assert torch.allclose(a.w_silu.float(), b.w_silu) and torch.allclose(a.c_spline.float(), b.c_spline)


def test_kan_univariate_in_a_grouped_static_tam():
    """kan(x) with one column and several groups fits and predicts (the column used to be read as a feature axis)."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(3):
        x = rng.uniform(-1, 1, size=200)
        rows.append(pd.DataFrame({"g": g, "t": np.arange(200), "x": x, "y": np.sin(3 * x) + 0.05 * rng.normal(size=200)}))
    df = pd.concat(rows, ignore_index=True)
    m = StaticTAM("y ~ kan(x, n_nodes=4, k_outer=6)", group_col="g", date_col="t").fit(df)
    pred = m.predict(df)["Estimatedy"].to_numpy()
    assert np.sqrt(np.mean((pred - df["y"].to_numpy()) ** 2)) < 0.25


def test_kan_reads_its_columns_out_of_range():
    """The columns a KAN effect reads are listed among the features that extrapolate."""
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([KANEffect("x", additional_features=["z"])]) == {"x", "z"}


def test_penalty_carries_a_small_ridge_on_its_null_space():
    """Each node block is D'D + 1e-6 I: the ridge keeps the penalty positive definite, so the REML log-determinant of the penalized subspace is defined."""
    effect = KANEffect("x", n_nodes=3, k_inner=5, k_outer=6, penalty_order=2)
    effect.initialize(torch.linspace(-1, 1, 60, dtype=torch.float64))
    block = effect.build_penalty_matrix()[:6, :6]
    assert float(torch.linalg.eigvalsh(block).min()) > 0.5e-6 * effect.lambda_p
