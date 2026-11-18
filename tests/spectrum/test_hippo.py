# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._hippo.HiPPOEffect``."""

import numpy as np
import pandas as pd
import pytest
import torch

import tam
from tam.model.additive import StaticTAM
from tam.model.spectrum import HiPPOEffect


def test_hippo_contract(normalized, penalty_shape):
    """Verifies coefficient counts, feature map shapes, and positive-definite penalty."""
    k = 12
    effect = HiPPOEffect("x", k=k, measure="legt", dt=0.05, discretization="bilinear", degree_pen=1.0)
    assert effect.get_n_coeffs() == k

    phi = effect.build_feature_map(normalized(4, 16))
    assert phi.shape == (4, 16, k)
    assert torch.isfinite(phi).all()

    p_shape = penalty_shape(effect)
    assert p_shape == (k, k)

    p_mat = effect.build_penalty_matrix()
    if p_mat.is_sparse:
        p_mat = p_mat.to_dense()
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all(), "Penalty matrix must be strictly positive-definite"


def test_hippo_fft_vs_recurrence_equivalence():
    """Verifies that causal FFT convolution matches sequential recurrence to numerical precision."""
    k = 8
    t_len = 64
    effect = HiPPOEffect("x", k=k, measure="legt", dt=0.02, discretization="bilinear")

    # Test in float64
    u = torch.sin(torch.linspace(0.0, 5.0, t_len, dtype=torch.float64))

    # Sequential recurrence reference
    a_bar = effect.a_bar.to(dtype=torch.float64)
    b_bar = effect.b_bar.to(dtype=torch.float64)
    s_rec = torch.zeros(t_len, k, dtype=torch.float64)
    curr_s = torch.zeros(k, dtype=torch.float64)
    for t in range(t_len):
        curr_s = a_bar @ curr_s + b_bar * u[t]
        s_rec[t] = curr_s

    # FFT evaluation
    phi_fft = effect._compute_ssm_states(u.unsqueeze(0), initial_state=None, use_fft=True)[0]
    max_diff = torch.max(torch.abs(s_rec - phi_fft)).item()
    assert max_diff < 1e-10, f"FFT and recurrence diff ({max_diff}) exceeds float64 tolerance"


def test_hippo_polynomial_reconstruction():
    """Verifies that the HiPPO Legendre basis reconstructs a smooth polynomial with RMSE < 1e-3."""
    k = 16
    t_len = 200
    t_grid = torch.linspace(0.0, 1.0, t_len, dtype=torch.float64)
    u = t_grid**2 - 0.5 * t_grid + 0.2

    effect = HiPPOEffect("x", k=k, measure="legt", dt=0.005, discretization="bilinear")
    phi = effect.build_feature_map(u)

    # After initial transient (t >= 50), reconstruct u[t] via linear regression
    s_feat = phi[50:].to(torch.float64)
    y_target = u[50:].unsqueeze(-1)
    w = torch.linalg.lstsq(s_feat, y_target).solution
    u_pred = s_feat @ w
    rmse = float(torch.sqrt(torch.mean((u_pred - y_target)**2)))

    assert rmse < 1e-3, f"Polynomial reconstruction RMSE ({rmse}) exceeds threshold 1e-3"


def test_hippo_strict_causality():
    """Verifies that modifying future input values does not alter past state rows."""
    k = 10
    t_len = 50
    effect = HiPPOEffect("x", k=k, measure="legt", dt=0.05)

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    # Modify only the future after t=30
    u2[30:] = u2[30:] + 10.0

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Past rows 0..29 must be bit-identical
    np.testing.assert_allclose(
        phi1[:30].numpy(), phi2[:30].numpy(), rtol=1e-12, atol=1e-12
    )

    # Verify causal state continuation across fit(train) -> predict(test)
    eff_full = HiPPOEffect("x", k=k, measure="legt", dt=0.05)
    eff_split = HiPPOEffect("x", k=k, measure="legt", dt=0.05)
    phi_full = eff_full.build_feature_map(u1)
    phi_tr = eff_split.build_feature_map(u1[:30])
    phi_te = eff_split.build_feature_map(u1[30:], state_in=eff_split._state_out)
    np.testing.assert_allclose(phi_te.numpy(), phi_full[30:].numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("measure", ["legt", "legs", "lagt"])
@pytest.mark.parametrize("discretization", ["bilinear", "zoh"])
def test_hippo_measures_and_discretizations(measure, discretization):
    """Verifies combinations of HiPPO measures and discretization operators."""
    k = 6
    effect = HiPPOEffect("x", k=k, measure=measure, discretization=discretization, dt=0.05)
    u = torch.rand(30, dtype=torch.get_default_dtype())
    phi = effect.build_feature_map(u)
    assert phi.shape == (30, k)
    assert torch.isfinite(phi).all()


def test_hippo_static_tam_fit_predict_decompose():
    """Verifies StaticTAM integration with hippo() effect: fit, predict, and decompose."""
    np.random.seed(42)
    n = 120
    x = np.sin(np.linspace(0.0, 10.0, n))
    y = 0.8 * x + 0.1 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:90], "Y": y[:90]})
    df_test = pd.DataFrame({"X": x[90:], "Y": y[90:]})

    model = StaticTAM(formula="Y ~ hippo(X, k=8, measure='legt', dt=0.05)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    decomp = model.decompose_prediction(df_test)
    effect_cols = [c for c in decomp.columns if c.startswith("effect_")]
    assert len(effect_cols) > 0
    assert any("X" in c for c in effect_cols)


def test_hippo_tensor_product_interaction():
    """Verifies that hippo() works inside te() tensor products."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = np.sin(x1) * np.cos(x2) + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ te(hippo(X1, k=6), s(X2, k=6))")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def test_hippo_state_continues_across_the_fft_regime():
    """Long passes use the FFT convolution: a split at any point still equals the single pass."""
    g = torch.Generator().manual_seed(0)
    u = torch.rand(3, 300, dtype=torch.float64, generator=g)
    eff = HiPPOEffect("x", k=8, measure="legs", dt=0.05, discretization="zoh")
    whole = eff.build_feature_map(u)
    first = eff.build_feature_map(u[:, :170])
    second = eff.build_feature_map(u[:, 170:], state_in=eff._state_out)
    assert torch.allclose(torch.cat([first, second], dim=1), whole, atol=1e-10)
    assert eff._state_out["c"].shape == (3, 8)
    for b in range(3):
        assert torch.allclose(eff.build_feature_map(u[b:b + 1])[0], whole[b], atol=1e-10)


def test_hippo_float32_input_stays_float32():
    eff = HiPPOEffect("x", k=6)
    out = eff.build_feature_map(torch.rand(2, 120, dtype=torch.float32))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_hippo_fifty_thousand_steps():
    import time

    eff = HiPPOEffect("x", k=8)
    t0 = time.time()
    out = eff.build_feature_map(torch.rand(1, 50_000, dtype=torch.float64))
    assert out.shape == (1, 50_000, 8) and torch.isfinite(out).all() and time.time() - t0 < 30.0


def test_hippo_refuses_isolated_point_extrapolation():
    with pytest.raises(ValueError, match="temporal effect"):
        HiPPOEffect("x", extrapolate="linear")


def test_hippo_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([HiPPOEffect("x")]) == {"x"}
