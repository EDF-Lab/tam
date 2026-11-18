# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for SingleIndexEffect and distributed lag smoothing."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.additive import StaticTAM
from tam.model.spectrum import SingleIndexEffect


def test_single_index_lag_causality():
    """Verifies that future rows do not alter past single-index lag representations."""
    effect = SingleIndexEffect("x", lag_window=10, k=8)
    t_len = 80

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    u2[50:] = u2[50:] + 10.0

    effect.initialize(u1)
    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Past rows 0..49 must be bit-identical
    np.testing.assert_allclose(
        phi1[:50].numpy(), phi2[:50].numpy(), rtol=1e-12, atol=1e-12
    )


def test_single_index_state_continuation():
    """Verifies that fit on [0, n) and predict on [n, m) carries lag history to match [0, m)."""
    torch.manual_seed(42)
    n = 100
    m = 160
    L = 8
    x_full = torch.randn(m, dtype=torch.float64)
    x_train = x_full[:n]
    x_test = x_full[n:]

    effect_single = SingleIndexEffect("x", lag_window=L, k=6)
    effect_split = SingleIndexEffect("x", lag_window=L, k=6)

    effect_single.initialize(x_train)
    effect_split.initialize(x_train)

    # 1. Full single pass
    phi_full = effect_single.build_feature_map(x_full)

    # 2. Split pass with state continuation
    phi_train = effect_split.build_feature_map(x_train)
    phi_train_repeat = effect_split.build_feature_map(x_train)
    np.testing.assert_allclose(phi_train.numpy(), phi_train_repeat.numpy(), rtol=1e-12, atol=1e-12)
    phi_test = effect_split.build_feature_map(x_test, state_in=effect_split._state_out)

    assert effect_split._state_out["hist"].shape == (1, L - 1, 1)
    assert effect_split.u_bounds is not None

    # phi_test must match phi_full[n:m] exactly
    np.testing.assert_allclose(
        phi_test.numpy(), phi_full[n:m].numpy(), rtol=1e-10, atol=1e-10
    )

    # Verify scale invariance on unnormalized physical covariate (e.g. Celsius in [-10, 40])
    x_phys = x_full * 15.0 + 15.0
    eff_phys = SingleIndexEffect("x", lag_window=L, k=6)
    eff_phys.initialize(x_phys[:n])
    phi_phys = eff_phys.build_feature_map(x_phys)
    assert (phi_phys.std(dim=0) > 1e-4).all()


def test_single_index_contract():
    """Verifies coefficient counts, output shapes, and penalty matrix."""
    k = 10
    effect = SingleIndexEffect("x", lag_window=12, k=k, lambda_p=2.0)

    n_coeffs = effect.get_n_coeffs()
    assert n_coeffs == k + 3  # degree 3

    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    phi = effect.build_feature_map(x)
    assert phi.shape == (50, n_coeffs)
    assert torch.isfinite(phi).all()

    p_mat = effect.build_penalty_matrix()
    assert p_mat.shape == (n_coeffs, n_coeffs)
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals >= -1e-10).all()


def test_single_index_multivariate():
    """Verifies multivariate single-index block."""
    k = 8
    effect = SingleIndexEffect("x1", additional_features=["x2", "x3"], k=k)
    assert effect.multi_input is True

    x_multi = torch.randn(40, 3, dtype=torch.float64)
    phi = effect.build_feature_map(x_multi)
    assert phi.shape == (40, k + 3)
    assert torch.isfinite(phi).all()


def test_static_tam_integration_single_index():
    """Verifies StaticTAM fit, predict, and decompose with si()."""
    np.random.seed(42)
    n = 120
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.sin(x1 * np.pi)
    y = 0.5 * x1 + 0.3 * (x1 + x2)**2 + 0.1 * np.random.randn(n)

    df_train = pd.DataFrame({"X1": x1[:80], "X2": x2[:80], "Y": y[:80]})
    df_test = pd.DataFrame({"X1": x1[80:], "X2": x2[80:], "Y": y[80:]})

    model = StaticTAM(formula="Y ~ s(X1, k=6) + si(X1, lag_window=6, k=6)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    assert any("si" in c or "single_index" in c for c in decomp.columns)


def test_single_index_multivariate_block_is_stateless_and_group_independent():
    """The multivariate block reads one row at a time: groups never mix and no state is carried."""
    effect = SingleIndexEffect("x1", additional_features=["x2"], k=6)
    assert effect.stateful is False
    g = torch.Generator().manual_seed(5)
    x = torch.randn(3, 60, 2, dtype=torch.float64, generator=g)
    effect.initialize(x)
    whole = effect.build_feature_map(x)
    assert whole.shape == (3, 60, 6 + 3)
    for b in range(3):
        assert torch.allclose(effect.build_feature_map(x[b:b + 1])[0], whole[b], atol=1e-12)


def test_single_index_statistics_are_frozen_at_initialize():
    """A later call on other data does not move the standardization, alpha or the index bounds."""
    effect = SingleIndexEffect("x1", additional_features=["x2"], k=6)
    x = torch.randn(2, 80, 2, dtype=torch.float64)
    effect.initialize(x)
    bounds, mean, alpha = effect.u_bounds, effect._x_mean.clone(), effect.alpha.copy()
    effect.build_feature_map(x * 7.0 + 3.0)
    assert effect.u_bounds == bounds and torch.equal(effect._x_mean, mean) and np.array_equal(effect.alpha, alpha)


def test_single_index_in_static_tam_with_groups():
    """Distributed-lag index inside a grouped StaticTAM: the end-of-training state has one row per group."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(3):
        x = np.cumsum(rng.normal(size=120)) * 0.1
        y = np.tanh(x) + 0.05 * rng.normal(size=120)
        rows.append(pd.DataFrame({"g": g, "t": np.arange(120), "x": x, "y": y}))
    df = pd.concat(rows, ignore_index=True)
    m = StaticTAM("y ~ si(x, lag_window=6, k=6)", group_col="g", date_col="t").fit(df)
    idx = [i for i, e in enumerate(m.effects_list_) if isinstance(e, SingleIndexEffect)][0]
    assert m._effect_states_[idx]["hist"].shape == (3, 5, 1)


def test_single_index_reads_its_columns_out_of_range():
    """The columns a single-index effect reads are listed among the features that extrapolate."""
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([SingleIndexEffect("x", additional_features=["z"])]) == {"x", "z"}


def test_single_index_reads_a_batch_of_one_column_as_b_t():
    """(B, T) is a batch of series for one column; (N, 1) rows and (T,) are one series."""
    effect = SingleIndexEffect("x", lag_window=5, k=5)
    x = torch.randn(2, 80, dtype=torch.float64)
    effect.initialize(x)
    assert effect.build_feature_map(x).shape == (2, 80, 5 + 3)
    assert torch.allclose(effect.build_feature_map(x), effect.build_feature_map(x.unsqueeze(-1)))
    assert effect.build_feature_map(x[0]).shape == (80, 5 + 3) and effect.build_feature_map(x[0].unsqueeze(-1)).shape == (80, 5 + 3)
