# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for TJEPAEffect (Temporal Joint-Embedding Predictive Architecture)."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.additive import StaticTAM
from tam.model.spectrum import TJEPAEffect


def test_tjepa_contract():
    """Verifies coefficient counts, output shapes, and penalty matrices for linear and spline readouts."""
    d = 3
    w = 12
    k_spline = 6

    # 1. Linear readout
    effect_lin = TJEPAEffect("x1", additional_features=["x2"], latent_dim=d, window=w, readout="linear", lambda_p=1.5)
    assert effect_lin.get_n_coeffs() == d
    p_lin = effect_lin.build_penalty_matrix()
    assert p_lin.shape == (d, d)
    assert torch.allclose(p_lin, torch.eye(d) * 1.5)

    # 2. Spline readout
    effect_spl = TJEPAEffect("x1", additional_features=["x2"], latent_dim=d, window=w, readout="spline", k_spline=k_spline, lambda_p=1.0)
    expected_coeffs = d * k_spline
    assert effect_spl.get_n_coeffs() == expected_coeffs
    p_spl = effect_spl.build_penalty_matrix()
    assert p_spl.shape == (expected_coeffs, expected_coeffs)
    eigvals = torch.linalg.eigvalsh(p_spl)
    assert (eigvals >= -1e-10).all()


def test_tjepa_causality():
    """Verifies that future rows do not alter past T-JEPA representations."""
    torch.manual_seed(42)
    effect = TJEPAEffect("x1", additional_features=["x2"], latent_dim=2, window=10, readout="linear")
    n = 100
    u1 = torch.rand(n, 2, dtype=torch.float64)
    u2 = u1.clone()
    u2[60:] = u2[60:] + 5.0

    # Fit encoder on common initial window
    effect.initialize(u1[:60])

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Past rows 0..59 must be bit-identical
    np.testing.assert_allclose(
        phi1[:60].numpy(), phi2[:60].numpy(), rtol=1e-12, atol=1e-12
    )


def test_tjepa_state_continuation():
    """Verifies that fit on [0, n) and predict on [n, m) carries sliding-window history."""
    torch.manual_seed(42)
    n = 120
    m = 180
    w = 12
    x_full = torch.randn(m, 2, dtype=torch.float64)
    x_train = x_full[:n]
    x_test = x_full[n:]

    effect_full = TJEPAEffect("x1", additional_features=["x2"], latent_dim=2, window=w, readout="linear")
    effect_split = TJEPAEffect("x1", additional_features=["x2"], latent_dim=2, window=w, readout="linear")

    effect_full.initialize(x_train)
    effect_split.initialize(x_train)

    phi_full = effect_full.build_feature_map(x_full)

    phi_train = effect_split.build_feature_map(x_train)
    phi_test = effect_split.build_feature_map(x_test, state_in=effect_split._state_out)

    assert effect_split._state_out["hist"].shape == (1, w - 1, 2)

    # phi_test must match phi_full[n:m] exactly
    np.testing.assert_allclose(
        phi_test.numpy(), phi_full[n:m].numpy(), rtol=1e-10, atol=1e-10
    )


def test_tjepa_noise_filtering_and_anti_collapse():
    """Verifies that T-JEPA filters uncorrelated observation noise and preserves full-rank latent variance."""
    torch.manual_seed(42)
    T = 800
    t = torch.linspace(0, 24, T, dtype=torch.float64)
    u1 = torch.sin(t)
    u2 = torch.cos(0.5 * t)
    noise = torch.randn(T, 4, dtype=torch.float64) * 0.5
    X = torch.cat([u1.unsqueeze(-1), u2.unsqueeze(-1), noise], dim=-1)

    effect = TJEPAEffect(
        "x1",
        additional_features=["x2", "x3", "x4", "x5", "x6"],
        latent_dim=2,
        window=16,
        horizon=4,
        readout="linear",
        epochs=35,
        mode="vicreg",
        seed=42,
    )
    phi = effect.build_feature_map(X)
    assert phi.shape == (T, 2)
    assert torch.isfinite(phi).all()

    # 1. Anti-collapse check: variance of each coordinate is active
    cov = torch.cov(phi.T)
    var_diag = torch.diagonal(cov)
    assert (var_diag > 0.05).all()

    # 2. Noise-filtering check: latent space correlates with true underlying slow drivers
    y_signal = (2.0 * u1 + 1.5 * u2).unsqueeze(-1)
    sol = torch.linalg.lstsq(phi, y_signal).solution
    pred = phi @ sol
    r2 = 1.0 - torch.mean((y_signal - pred) ** 2) / torch.var(y_signal)
    assert r2.item() > 0.85


def test_tjepa_statictam_integration():
    """Verifies StaticTAM parsing, fit, and predict with tjepa() term."""
    np.random.seed(42)
    n = 150
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.cos(x1 * np.pi)
    x3 = np.sin(x1 * np.pi)
    y = 0.8 * x1 + 0.4 * x2 + 0.1 * np.random.randn(n)

    df_train = pd.DataFrame({"X1": x1[:100], "X2": x2[:100], "X3": x3[:100], "Y": y[:100]})
    df_test = pd.DataFrame({"X1": x1[100:], "X2": x2[100:], "X3": x3[100:], "Y": y[100:]})

    model = StaticTAM(
        formula="Y ~ s(X1, k=6) + tjepa(X1, others='X2|X3', latent_dim=2, window=12, readout='linear')"
    )
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    assert any("tjepa" in c for c in decomp.columns)


def test_tjepa_univariate_batch_layout_and_group_windows():
    """One column read as (B, T) gives (B, T, K); training windows never span two groups."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 500, dtype=torch.float64, generator=g)
    eff = TJEPAEffect("x", latent_dim=2, window=8, horizon=2, readout="linear", epochs=5, mode="vicreg")
    eff.initialize(x.unsqueeze(-1))
    phi = eff.build_feature_map(x)
    assert phi.shape == (2, 500, 2) and torch.isfinite(phi).all()
    assert torch.allclose(phi, eff.build_feature_map(x.unsqueeze(-1)))
    # a group whose values are all 1e6 never meets the other group inside a window: the other group's features do not move
    huge = torch.stack([x[0], torch.full((500,), 1e6, dtype=torch.float64)])
    eff2 = TJEPAEffect("x", latent_dim=2, window=8, horizon=2, readout="linear", epochs=5, mode="vicreg")
    eff2.initialize(huge.unsqueeze(-1))
    alone = eff2.build_feature_map(huge[:1])[0]
    stacked = eff2.build_feature_map(huge)[0]
    assert torch.allclose(alone, stacked, atol=1e-10)


def test_tjepa_needs_enough_training_rows_per_group():
    eff = TJEPAEffect("x", window=16, horizon=4)
    with pytest.raises(ValueError, match="horizon training rows per group"):
        eff.initialize(torch.randn(3, 10, 1, dtype=torch.float64))


def test_tjepa_does_not_touch_the_global_random_state():
    x = torch.randn(2, 200, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(7)
    before = torch.rand(3)
    torch.manual_seed(7)
    eff = TJEPAEffect("x", latent_dim=2, window=6, horizon=2, readout="linear", epochs=5, mode="vicreg")
    eff.initialize(x)
    assert torch.equal(torch.rand(3), before)


def test_tjepa_penalty_weights_follow_the_latent_predictor_residual():
    """Each latent block is weighted by 1 + the relative error of the latent predictor; before the fit every weight is 1."""
    eff = TJEPAEffect("x", latent_dim=3, window=8, horizon=2, readout="spline", k_spline=5, epochs=8, mode="vicreg", lambda_p=2.0)
    P0 = eff.build_penalty_matrix()
    D = torch.diff(torch.eye(5, dtype=P0.dtype), n=2, dim=0)
    assert torch.allclose(P0, torch.kron(torch.eye(3, dtype=P0.dtype), 2.0 * (D.T @ D)))
    g = torch.Generator().manual_seed(1)
    eff.initialize(torch.cumsum(torch.randn(2, 300, 1, dtype=torch.float64, generator=g), 1))
    assert eff._latent_res.shape == (3,) and bool((eff._latent_res >= 0).all())
    P = eff.build_penalty_matrix()
    for j in range(3):
        block = P[5 * j:5 * (j + 1), 5 * j:5 * (j + 1)]
        assert torch.allclose(block, (1.0 + eff._latent_res[j]).to(P.dtype) * 2.0 * (D.T @ D))
    assert float(torch.linalg.eigvalsh(P).min()) >= -1e-10 and torch.allclose(P, P.T)
    lin = TJEPAEffect("x", latent_dim=3, window=8, horizon=2, readout="linear", epochs=8, mode="vicreg")
    lin.initialize(torch.cumsum(torch.randn(2, 300, 1, dtype=torch.float64, generator=g), 1))
    assert bool((torch.diagonal(lin.build_penalty_matrix()) >= 1.0).all())


def test_tjepa_refuses_isolated_point_extrapolation():
    with pytest.raises(ValueError, match="temporal effect"):
        TJEPAEffect("x", extrapolate="linear")


def test_tjepa_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([TJEPAEffect("x", additional_features=["z"])]) == {"x", "z"}


def _series(b=2, t=200, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.sin(torch.arange(t, dtype=torch.float64) / 5.0)
    return torch.stack([torch.stack([base * (1 + 0.3 * i) + 0.1 * torch.randn(t, dtype=torch.float64, generator=g) for _ in range(d)], dim=-1) for i in range(b)])


def test_svd_mode_is_the_default_and_deterministic():
    """The closed-form encoder is the default; two fits give bit-identical features."""
    x = _series()
    a = TJEPAEffect("x", additional_features=["z"], latent_dim=3, window=8, horizon=2)
    b = TJEPAEffect("x", additional_features=["z"], latent_dim=3, window=8, horizon=2)
    assert a.mode == "svd"
    a.initialize(x)
    b.initialize(x)
    assert torch.equal(a.build_feature_map(x), b.build_feature_map(x))
    assert torch.equal(a._latent_res, b._latent_res)
    assert (a._latent_res >= 0).all()


def test_modes_are_validated_and_both_fit():
    """Any other mode raises; the gradient encoder stays available."""
    with pytest.raises(ValueError, match="mode"):
        TJEPAEffect("x", mode="pca")
    x = _series(b=1, t=120)
    for mode in ("svd", "vicreg"):
        eff = TJEPAEffect("x", additional_features=["z"], latent_dim=2, window=6, horizon=2, epochs=5, mode=mode)
        eff.initialize(x)
        phi = eff.build_feature_map(x)
        assert phi.shape == (1, 120, eff.get_n_coeffs()) and torch.isfinite(phi).all()
        assert eff.build_penalty_matrix().shape == (eff.get_n_coeffs(),) * 2


def test_svd_encoder_features_do_not_depend_on_the_other_group():
    """The encoder is fit once; the features of a group depend on its own rows only."""
    x = _series(b=2, t=100)
    eff = TJEPAEffect("x", additional_features=["z"], latent_dim=2, window=6, horizon=2)
    eff.initialize(x)
    assert torch.allclose(eff.build_feature_map(x)[:1], eff.build_feature_map(x[:1]), atol=1e-12)


def test_formula_reads_the_mode():
    """`tjepa(x, mode='vicreg')` selects the gradient encoder; the default is the closed-form one."""
    from tam import StaticTAM

    for spec, expected in (("", "svd"), (", mode='vicreg'", "vicreg"), (", mode='svd'", "svd")):
        model = StaticTAM(formula=f"y ~ tjepa(x{spec})", group_col="g", date_col="d")
        assert [e for e in model.effects_list_ if e.feature_name == "x"][0].mode == expected


def test_svd_mode_is_fast():
    """The closed-form fit does not iterate: well under 2 seconds for 5000 rows in 2 groups."""
    import time

    x = _series(b=2, t=2500)
    eff = TJEPAEffect("x", additional_features=["z"], latent_dim=4, window=16, horizon=4)
    t0 = time.time()
    eff.initialize(x)
    assert time.time() - t0 < 2.0
