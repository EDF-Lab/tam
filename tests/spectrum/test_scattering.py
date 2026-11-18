# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the Wavelet Scattering Transform effect (wst)."""

import math
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum._scattering import ScatteringEffect


def test_scattering_channel_count_and_positivity():
    """Checks exact column count 1 + J*Q + (J*Q choose 2) and envelope non-negativity."""
    J, Q = 3, 1
    N1 = J * Q
    expected_channels = 1 + N1 + (N1 * (N1 - 1)) // 2  # 1 + 3 + 3 = 7

    eff = ScatteringEffect("temp", J=J, Q=Q, max_order=2, causal=True)
    assert eff._n_coeffs == expected_channels

    # Verify Morlet wavelets satisfy exact discrete zero-mean admissibility sum(psi) == 0
    for re_f, im_f in zip(eff.psi_real, eff.psi_imag):
        assert abs(float(re_f.sum().item())) < 1e-6, f"Non-zero mean in real Morlet filter: {re_f.sum().item()}"
        assert abs(float(im_f.sum().item())) < 1e-6, f"Non-zero mean in imag Morlet filter: {im_f.sum().item()}"

    x = torch.randn(100, dtype=torch.float32)
    phi = eff.build_feature_map(x)
    assert phi.shape == (100, expected_channels)

    # Order 1 and Order 2 channels are smoothed modulus envelopes, hence >= 0
    s1_and_s2 = phi[:, 1:]
    assert torch.all(s1_and_s2 >= -1e-6)

    # Penalty matrix check
    P = eff.build_penalty_matrix()
    assert P.shape == (expected_channels, expected_channels)
    assert torch.all(torch.diag(P) > 0)


def test_scattering_causality():
    """Modifying future observations at s > t leaves row t strictly unchanged."""
    eff = ScatteringEffect("temp", J=3, Q=1, max_order=2, causal=True)

    torch.manual_seed(42)
    x1 = torch.randn(80, dtype=torch.float32)
    x2 = x1.clone()
    # Modify future rows starting from index 40
    x2[40:] = x2[40:] + 10.0

    phi1 = eff.build_feature_map(x1)
    phi2 = eff.build_feature_map(x2)

    # Rows 0..39 must be identical to numerical tolerance
    diff = torch.abs(phi1[:40] - phi2[:40]).max().item()
    assert diff < 1e-6, f"Causality violated: max past difference was {diff}"


def test_scattering_state_continuation():
    """State continuation across fit -> predict boundary matches unified pass."""
    eff = ScatteringEffect("temp", J=3, Q=1, max_order=2, causal=True)

    torch.manual_seed(123)
    x = torch.randn(150, dtype=torch.float32)

    # Unified single pass
    phi_unified = eff.build_feature_map(x)

    # Two-stage pass
    phi_train = eff.build_feature_map(x[:100])
    phi_test = eff.build_feature_map(x[100:], state_in=eff._state_out)

    diff_train = torch.abs(phi_train - phi_unified[:100]).max().item()
    diff_test = torch.abs(phi_test - phi_unified[100:]).max().item()

    assert diff_train < 1e-6, f"Train mismatch: {diff_train}"
    assert diff_test < 1e-5, f"Test state continuation mismatch: {diff_test}"


def test_scattering_deformation_stability():
    """WST envelope shows Lipschitz stability under time-warping phase shifts."""
    eff = ScatteringEffect("pulse", J=3, Q=1, max_order=2, causal=False)

    t = torch.linspace(-5, 5, 200)
    x = torch.exp(-0.5 * (t / 0.5) ** 2)  # localized Gaussian pulse

    # Small time shift by 2 time steps
    x_shifted = torch.roll(x, shifts=2)

    phi = eff.build_feature_map(x)
    phi_shifted = eff.build_feature_map(x_shifted)

    # Relative L2 distance on scattering envelopes
    rel_change = torch.norm(phi - phi_shifted) / (torch.norm(phi) + 1e-8)
    assert rel_change.item() < 0.6  # Stable bounded perturbation


def test_scattering_static_tam_integration():
    """Full StaticTAM fit and predict with wst() effect in formula."""
    T = 60
    dates = pd.date_range("2022-01-01", periods=T, freq="h")
    df = pd.DataFrame({
        "date": dates,
        "tod": [d.hour for d in dates],
        "Load": np.sin(np.linspace(0, 4 * np.pi, T)) * 1000 + 50000,
        "temperature": np.cos(np.linspace(0, 2 * np.pi, T)) * 15 + 10,
    })

    model = ta.StaticTAM("Load ~ wst(temperature, J=3, Q=1, ap=-5)", group_col="tod", date_col="date")
    model.fit(df)

    preds = model.predict(df)
    assert "EstimatedLoad" in preds.columns
    assert len(preds) == T
    assert not preds["EstimatedLoad"].isna().any()

    # Decompose prediction check
    terms = model.decompose_prediction(df)
    assert any("wst" in c or "temperature" in c for c in terms.columns)


def test_scattering_penalty_couples_the_octaves_of_the_first_order():
    """Order 0 and order 2 keep their diagonal; the order-1 block gains the second-difference coupling D2'D2 along the octaves."""
    lam = 2.0
    eff = ScatteringEffect("x", J=4, Q=1, max_order=2, lambda_p=lam)
    P = eff.build_penalty_matrix()
    n1 = eff.N1
    dtype = P.dtype
    base = torch.diag(torch.tensor([1.0] + [eff.order_pen * (s / eff.xi_0) ** 2 for s in eff.scales]
                                   + [eff.order_pen ** 2 * ((eff.scales[a] / eff.xi_0) ** 2 + (eff.scales[b] / eff.xi_0) ** 2)
                                      for a in range(n1) for b in range(a + 1, n1)], dtype=dtype))
    D2 = torch.diff(torch.eye(n1, dtype=dtype), n=2, dim=0)
    expected = base.clone()
    expected[1:1 + n1, 1:1 + n1] += D2.T @ D2
    assert torch.allclose(P, lam * expected)
    assert float(torch.linalg.eigvalsh(P).min()) > 0 and torch.allclose(P, P.T)
    assert P[0, 1:].abs().sum() == 0 and P[1 + n1:, :1 + n1].abs().sum() == 0 and P[1:1 + n1, 1 + n1:].abs().sum() == 0
    few = ScatteringEffect("x", J=2, Q=1, max_order=2, lambda_p=lam)           # fewer than three octaves: no coupling
    assert torch.count_nonzero(few.build_penalty_matrix() - torch.diag(torch.diagonal(few.build_penalty_matrix()))) == 0


def test_scattering_filters_are_zero_mean_in_the_dtype_of_the_input():
    eff = ScatteringEffect("x", J=3, Q=2, max_order=2)
    for dtype in (torch.float32, torch.float64):
        phi, re, im = eff._filters(torch.device("cpu"), dtype)
        assert phi.dtype == dtype and all(abs(float(f.sum())) < 1e-6 for f in re + im)
        if dtype == torch.float64:
            assert all(abs(float(f.sum())) < 1e-12 for f in re + im)


def test_scattering_groups_and_float64():
    eff = ScatteringEffect("x", J=3, Q=1, max_order=2, causal=True)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 300, dtype=torch.float64, generator=g)
    whole = eff.build_feature_map(x)
    assert whole.dtype == torch.float64 and whole.shape == (3, 300, eff.get_n_coeffs())
    assert torch.allclose(eff.build_feature_map(x[1:2])[0], whole[1], atol=1e-12)
    one = eff.build_feature_map(x[:, :1])                              # a single step is computed, not zero
    assert one.shape == (3, 1, eff.get_n_coeffs()) and float(one.abs().sum()) > 0
    sym = ScatteringEffect("x", J=2, Q=1, max_order=1, causal=False)
    assert sym.stateful is False and torch.allclose(sym.build_feature_map(x[2:3])[0], sym.build_feature_map(x)[2], atol=1e-12)


def test_scattering_refuses_isolated_point_extrapolation_when_causal():
    with pytest.raises(ValueError, match="temporal effect"):
        ScatteringEffect("x", extrapolate="linear")
    ScatteringEffect("x", extrapolate="linear", causal=False)


def test_scattering_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([ScatteringEffect("x")]) == {"x"}


def test_formula_reads_the_carrier_frequency():
    """`wst(x, xi_0=2.5)` sets the carrier frequency of the mother wavelet; the default stays 0.85 pi."""
    import math

    from tam import StaticTAM

    model = StaticTAM(formula="y ~ wst(x, J=3, xi_0=2.5)", group_col="g", date_col="d")
    assert [e for e in model.effects_list_ if e.feature_name == "x"][0].xi_0 == 2.5
    model = StaticTAM(formula="y ~ wst(x, J=3)", group_col="g", date_col="d")
    assert [e for e in model.effects_list_ if e.feature_name == "x"][0].xi_0 == 0.85 * math.pi
