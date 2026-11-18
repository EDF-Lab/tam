# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import StaticTAM
from tam.model.spectrum._inertia import DegreeDayEffect
from tam.model.spectrum._inertia import InertiaEffect

def test_degree_day_causality():
    """Verifies that degree day hinges are strictly point-wise and causal."""
    effect = DegreeDayEffect("temp", heat_thresholds=(-0.2, 0.0), cool_thresholds=(0.3, 0.5), quad=True)
    t_len = 80

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    u2[50:] = u2[50:] + 5.0

    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    np.testing.assert_allclose(
        phi1[:50].numpy(), phi2[:50].numpy(), rtol=1e-12, atol=1e-12
    )



def test_degree_day_contract():
    """Verifies output shapes, coefficient counts, and positive-definite penalty."""
    heat = (-0.3, -0.1, 0.0)
    cool = (0.2, 0.4)
    effect_quad = DegreeDayEffect("temp", heat_thresholds=heat, cool_thresholds=cool, quad=True, lambda_p=2.0)
    effect_no_quad = DegreeDayEffect("temp", heat_thresholds=heat, cool_thresholds=cool, quad=False, lambda_p=2.0)

    # With quad: 2 * (3 + 2) = 10
    assert effect_quad.get_n_coeffs() == 10
    # Without quad: 3 + 2 = 5
    assert effect_no_quad.get_n_coeffs() == 5

    x = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64)
    phi_quad = effect_quad.build_feature_map(x)
    assert phi_quad.shape == (60, 10)
    assert torch.isfinite(phi_quad).all()

    p_mat = effect_quad.build_penalty_matrix()
    assert p_mat.shape == (10, 10)
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all(), "Ridge penalty must be strictly positive-definite"



def test_static_tam_integration_thermal_and_degree_day():
    """Verifies StaticTAM model fitting and prediction with thermal() and hdd_cdd()."""
    np.random.seed(42)
    n = 150
    temp = np.linspace(-10.0, 35.0, n)
    toy = np.linspace(0.0, 365.0, n)
    # Synthetic heating response
    y = np.maximum(0.0, 15.0 - temp) * 100.0 + 50.0 * np.sin(2 * np.pi * toy / 365.0) + np.random.randn(n) * 10.0

    df_train = pd.DataFrame({"temp": temp[:100], "toy": toy[:100], "load": y[:100]})
    df_test = pd.DataFrame({"temp": temp[100:], "toy": toy[100:], "load": y[100:]})

    formula = "load ~ s(toy, k=8) + thermal(temp, alphas=0.05|0.01, k=6) + hdd_cdd(temp, sharpness=5.0, quad=True)"
    model = StaticTAM(formula=formula)
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(df_test)
    assert not preds["Estimatedload"].isna().any()

    # Decompose prediction
    decomp = model.decompose_prediction(df_test)
    assert any("thermal" in c or "inertia" in c for c in decomp.columns)
    assert any("hdd_cdd" in c or "degree_day" in c for c in decomp.columns)


def test_smooth_hinge_is_the_degree_day_effect_under_its_own_token():
    """`smooth_hinge`, `dd` and `hdd_cdd` build the same effect."""
    from tam.model.spectrum._inertia import DegreeDayEffect, SmoothHingeEffect

    assert SmoothHingeEffect is DegreeDayEffect
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"temp": rng.uniform(-1, 1, 120)})
    df["y"] = 2.0 * np.maximum(0.1 - df["temp"], 0) + 0.05 * rng.normal(size=120)
    preds = []
    for token in ("hdd_cdd", "dd", "smooth_hinge"):
        m = StaticTAM(f"y ~ {token}(temp, sharpness=8)").fit(df)
        assert isinstance(m.effects_list_[-1], DegreeDayEffect)
        preds.append(m.predict(df)["Estimatedy"].to_numpy())
    assert np.array_equal(preds[0], preds[1]) and np.array_equal(preds[0], preds[2])


def test_degree_day_penalty_weights_and_family_coupling():
    """Squared hinges weigh 4, the hinges of one family are coupled by a first difference, two families never mix."""
    lam = 3.0
    eff = DegreeDayEffect("t", heat_thresholds=(0.2, -0.1), cool_thresholds=(0.5, 0.3, 0.7), quad=True, lambda_p=lam)
    assert eff.heat_thresholds == (-0.1, 0.2) and eff.cool_thresholds == (0.3, 0.5, 0.7)    # columns follow the ascending thresholds
    P = eff.build_penalty_matrix()
    n = eff.get_n_coeffs()
    assert P.shape == (n, n) and torch.allclose(P, P.T)
    assert float(torch.linalg.eigvalsh(P).min()) >= -1e-10
    # heating: hinge columns 0 and 2, squared-hinge columns 1 and 3
    assert P[1, 1].item() == pytest.approx(4.0 * lam) and P[3, 3].item() == pytest.approx(4.0 * lam)
    assert P[0, 2].item() == pytest.approx(-1.0 * lam) and P[0, 0].item() == pytest.approx(2.0 * lam)
    assert P[1, 3].item() == 0.0 and P[0, 1].item() == 0.0
    # cooling: hinge columns 4, 6, 8; the end hinges are coupled once, the middle one twice
    assert [P[c, c].item() / lam for c in (4, 6, 8)] == pytest.approx([2.0, 3.0, 2.0])
    assert P[4, 6].item() == pytest.approx(-lam) and P[6, 8].item() == pytest.approx(-lam) and P[4, 8].item() == 0.0
    assert torch.count_nonzero(P[:4, 4:]).item() == 0                      # the two families never mix


def test_degree_day_single_threshold_per_side_has_no_coupling():
    eff = DegreeDayEffect("t", heat_thresholds=(0.0,), cool_thresholds=(0.5,), quad=True, lambda_p=2.0)
    assert torch.allclose(eff.build_penalty_matrix(), torch.diag(torch.tensor([2.0, 8.0, 2.0, 8.0], dtype=torch.get_default_dtype())))
    plain = DegreeDayEffect("t", heat_thresholds=(0.0, 0.2), cool_thresholds=(0.5,), quad=False, lambda_p=1.0)
    assert torch.allclose(plain.build_penalty_matrix()[:2, :2], torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.get_default_dtype()))


def test_degree_day_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([DegreeDayEffect("t")]) == {"t"}
