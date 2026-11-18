# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam import StaticTAM
from tam.model.spectrum._inertia import InertiaEffect, ThermalInertiaEffect

def test_thermal_inertia_causality():
    """Verifies that future rows do not alter past thermal inertia representations."""
    effect = ThermalInertiaEffect("temp", alphas=(0.05, 0.01), k=6)
    t_len = 100

    u1 = torch.rand(t_len, dtype=torch.float64)
    u2 = u1.clone()
    u2[60:] = u2[60:] + 15.0

    effect.initialize(u1)
    phi1 = effect.build_feature_map(u1)
    phi2 = effect.build_feature_map(u2)

    # Past rows 0..59 must be bit-identical
    np.testing.assert_allclose(
        phi1[:60].numpy(), phi2[:60].numpy(), rtol=1e-12, atol=1e-12
    )



def test_thermal_inertia_state_continuation():
    """Verifies state carry-over: fit on [0, n) and predict on [n, m) matches [0, m)."""
    torch.manual_seed(42)
    n = 120
    m = 200
    x_full = torch.randn(m, dtype=torch.float64)
    x_train = x_full[:n]
    x_test = x_full[n:]

    effect_single = ThermalInertiaEffect("temp", alphas=(0.05, 0.01, 0.002), k=6)
    effect_split = ThermalInertiaEffect("temp", alphas=(0.05, 0.01, 0.002), k=6)

    # Fit domain normalization bounds on training data for both effects
    effect_single.initialize(x_train)
    effect_split.initialize(x_train)

    # 1. Full single pass over [0, m)
    phi_full = effect_single.build_feature_map(x_full)

    # 2. Sequential pass: train on [0, n), then test on [n, m) with carried state
    phi_train = effect_split.build_feature_map(x_train)
    # Verify idempotence on repeated training call
    phi_train_2 = effect_split.build_feature_map(x_train)
    np.testing.assert_allclose(phi_train.numpy(), phi_train_2.numpy(), rtol=1e-12, atol=1e-12)

    phi_test = effect_split.build_feature_map(x_test, state_in=effect_split._state_out)

    # Check terminal state was recorded
    assert effect_split._state_out["y"].shape == (1, 3)

    # phi_test must match phi_full[n:m] to high numerical precision
    np.testing.assert_allclose(
        phi_test.numpy(), phi_full[n:m].numpy(), rtol=1e-10, atol=1e-10
    )

    # Verify scale invariance on unnormalized physical temperatures in [-15, 42] Celsius
    temp_phys = x_full * 15.0 + 12.0
    eff_phys = InertiaEffect("temp", inertia_type="thermal", alphas=(0.05, 0.01, 0.002), k=6)
    eff_phys.initialize(temp_phys[:n])
    phi_phys = eff_phys.build_feature_map(temp_phys)
    np.testing.assert_allclose(phi_phys.numpy(), phi_full.numpy(), rtol=1e-5, atol=1e-5)



def test_thermal_inertia_contract():
    """Verifies output shapes, coefficient counts, and penalty matrix."""
    k = 8
    alphas = (0.05, 0.01, 0.002)
    effect = ThermalInertiaEffect("temp", alphas=alphas, k=k, lambda_p=1.5)

    n_coeffs = effect.get_n_coeffs()
    assert n_coeffs == len(alphas) * (k + 3)

    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    phi = effect.build_feature_map(x)
    assert phi.shape == (50, n_coeffs)
    assert torch.isfinite(phi).all()

    p_mat = effect.build_penalty_matrix()
    assert p_mat.shape == (n_coeffs, n_coeffs)

    # Semi-definite second differences: eigenvalues non-negative
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals >= -1e-10).all()



def test_inertia_effect_type_thermal():
    """Verifies InertiaEffect class and inertia(type='thermal') formula interface."""
    effect = InertiaEffect("temp", inertia_type="thermal", alphas=(0.05, 0.01), k=6)
    assert effect.summary_type() == "Inertia[thermal]"
    assert isinstance(effect, ThermalInertiaEffect)

    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "temp": np.linspace(-5.0, 30.0, n),
        "y": np.linspace(100.0, 200.0, n) + np.random.randn(n),
    })

    formula = "y ~ inertia(temp, type='thermal', alphas=0.05|0.01, k=6, ap=-4)"
    model = StaticTAM(formula=formula)
    model.fit(df)
    preds = model.predict(df)
    assert not preds["Estimatedy"].isna().any()
    decomp = model.decompose_prediction(df)
    assert "effect_temp" in decomp.columns


def test_inertia_matches_the_iir_filter_reference_on_every_group():
    """The smoothed trajectories equal scipy's first-order filter started at the first input, in the loop and the FFT regimes."""
    import scipy.signal as sps

    from tam.model.spectrum._inertia import _cox_de_boor

    alphas = (0.3, 0.05, 0.002)
    for T in (40, 700):
        g = torch.Generator().manual_seed(T)
        x = torch.randn(3, T, dtype=torch.float64, generator=g) * 4.0 + 10.0
        eff = InertiaEffect("temp", alphas=alphas, k=6)
        eff.initialize(x)
        phi = eff.build_feature_map(x)
        low, high = eff._x_bounds
        knots = eff._build_knots(device=x.device, dtype=x.dtype)
        for b in range(3):
            xn = (2.0 * (x[b] - low) / (high - low) - 1.0).numpy()
            cols = []
            for a in alphas:
                y, _ = sps.lfilter([a], [1.0, -(1.0 - a)], xn, zi=np.array([(1.0 - a) * xn[0]]))
                cols.append(_cox_de_boor(torch.tensor(y), knots, 3))
            np.testing.assert_allclose(phi[b].numpy(), torch.cat(cols, dim=-1).numpy(), atol=1e-9)


def test_inertia_refuses_isolated_point_extrapolation():
    """A temporal effect has no finite difference on isolated points: linear and saturation are refused at construction."""
    for mode in ("linear", "saturation"):
        with pytest.raises(ValueError, match="temporal effect"):
            InertiaEffect("temp", extrapolate=mode)
    InertiaEffect("temp", extrapolate="constant")


def test_inertia_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([InertiaEffect("temp")]) == {"temp"}


def test_decomposition_names_prefix_of_every_inertia_type():
    """Two effects on one column are told apart by the short prefix `inertia` whatever the inertia type."""
    from tam.model._math import decomposition_names

    for kind in ("thermal", "rotational", "behavioral"):
        names = decomposition_names([InertiaEffect("temp", inertia_type=kind), InertiaEffect("temp", inertia_type="thermal", alphas=(0.3,))])
        assert names[0] == "inertia_temp", names
