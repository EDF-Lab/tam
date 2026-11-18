# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the Bispectrum Effect and Hinich Bispectral Diagnostics."""

import math
import numpy as np
import pandas as pd
import pytest
import torch

from tam.model.additive import StaticTAM
from tam.model.diagnostics import hinich_bispectrum_test
from tam.model.spectrum._bispectrum import BispectrumEffect


def test_hinich_bispectrum_qpc_detection():
    """Tests quadratic phase coupling (QPC) detection vs independent uncoupled harmonics.

    Signal A has uncoupled phases (bicoherence vanishes across segments).
    Signal B has coupled phase (phi_3 = phi_1 + phi_2, bicoherence approaches 1.0).
    """
    n_fft = 128
    n_segs = 60
    t_total = n_segs * n_fft
    t_axis = np.arange(t_total)

    # Fundamental frequencies (cycles per window)
    k1 = 12
    k2 = 18
    k3 = k1 + k2  # 30

    omega1 = 2 * np.pi * k1 / n_fft
    omega2 = 2 * np.pi * k2 / n_fft
    omega3 = 2 * np.pi * k3 / n_fft

    rng = np.random.RandomState(42)

    # Synthesize segment-by-segment to simulate stationary segments
    sig_a_segs = []
    sig_b_segs = []

    for _ in range(n_segs):
        t_seg = np.arange(n_fft)
        phi1 = rng.uniform(0, 2 * np.pi)
        phi2 = rng.uniform(0, 2 * np.pi)
        phi3_uncoupled = rng.uniform(0, 2 * np.pi)
        phi3_coupled = phi1 + phi2

        # Uncoupled signal segment
        s_a = (
            np.cos(omega1 * t_seg + phi1)
            + np.cos(omega2 * t_seg + phi2)
            + np.cos(omega3 * t_seg + phi3_uncoupled)
        )
        # Coupled signal segment
        s_b = (
            np.cos(omega1 * t_seg + phi1)
            + np.cos(omega2 * t_seg + phi2)
            + np.cos(omega3 * t_seg + phi3_coupled)
        )

        sig_a_segs.append(s_a)
        sig_b_segs.append(s_b)

    sig_a = np.concatenate(sig_a_segs)
    sig_b = np.concatenate(sig_b_segs)

    # Run Hinich bispectrum test on both signals
    diag_a = hinich_bispectrum_test(sig_a, n_fft=n_fft, top_k=5)
    diag_b = hinich_bispectrum_test(sig_b, n_fft=n_fft, top_k=5)

    # Check bicoherence at coupled coordinate (k2, k1) or (k1, k2)
    b2_a = diag_a["bicoherence_map"][k1, k2]
    b2_b = diag_b["bicoherence_map"][k1, k2]

    assert b2_a < 0.15, f"Expected uncoupled bicoherence < 0.15, got {b2_a:.4f}"
    assert b2_b > 0.85, f"Expected coupled bicoherence > 0.85, got {b2_b:.4f}"

    # Top triad of signal B should correspond to (k1, k2)
    top_triad = diag_b["top_triads"][0]
    triad_indices = sorted([top_triad["k1"], top_triad["k2"]])
    expected_indices = sorted([k1, k2])
    assert triad_indices == expected_indices, f"Expected top triad {expected_indices}, got {triad_indices}"


def test_hinich_bispectrum_gaussianity_null():
    """Tests that Hinich Gaussianity test does not falsely reject Gaussian white noise."""
    rng = np.random.RandomState(123)
    white_noise = rng.randn(4096)

    diag = hinich_bispectrum_test(white_noise, n_fft=128, top_k=4)

    assert diag["is_gaussian"] is True
    assert diag["p_value_gaussianity"] > 0.01, f"False rejection: p={diag['p_value_gaussianity']}"


def test_bispectrum_contract():
    """Tests shape, coefficient count, and positive-definite penalty matrix for BispectrumEffect."""
    k = 8
    sobolev_order = 1.5
    effect = BispectrumEffect(
        feature_name="x",
        k=k,
        n_fft=128,
        sobolev_order=sobolev_order,
        lambda_p=2.0,
    )

    assert effect.get_n_coeffs() == 4 * k
    assert effect.summary_details() == f"k={k}, Sobolev(s={sobolev_order})"
    assert effect.penalty_coordinates() == [2.0]

    # Test penalty coordinates update
    effect.set_penalty_coordinates([5.0])
    assert effect.penalty_coordinates() == [5.0]

    # Test penalty matrix properties
    p_mat = effect.build_penalty_matrix()
    assert p_mat.shape == (4 * k, 4 * k)
    # Must be strictly positive definite
    eigenvals = torch.linalg.eigvalsh(p_mat)
    assert torch.all(eigenvals > 0.0), "Penalty matrix P must be strictly positive definite"

    # Test feature map shape, full column rank, and zero constant columns
    x_test = torch.linspace(-1.0, 1.0, 120, dtype=torch.float64)
    phi = effect.build_feature_map(x_test)
    assert phi.shape == (120, 4 * k)
    assert not torch.isnan(phi).any()
    assert (phi.std(dim=0) > 1e-4).all(), "Bispectrum columns must be strictly non-constant"
    rank_phi = int(torch.linalg.matrix_rank(phi).item())
    assert rank_phi >= 4 * k - 3, f"Expected nearly full column rank {4 * k}, got {rank_phi}"   # a smooth ramp is a poor signal for the rank


def test_bispectrum_manual_triads():
    """Tests that explicit user-specified triads are respected."""
    triads = [(2, 3), (4, 5)]
    effect = BispectrumEffect(
        feature_name="x",
        k=2,
        n_fft=64,
        triads=triads,
    )

    assert effect.get_n_coeffs() == 4 * 2
    assert len(effect.triads) == 2
    assert effect.triads == triads

    x = torch.linspace(-1.0, 1.0, 50)
    phi = effect.build_feature_map(x)
    assert phi.shape == (50, 8)


def test_bispectrum_static_tam_fit_predict_decompose():
    """Tests full integration of bispec() inside StaticTAM."""
    n_samples = 200
    t_vals = np.linspace(0, 10, n_samples)
    y_vals = np.sin(t_vals) + 0.5 * np.sin(2 * t_vals) + 0.1 * np.random.randn(n_samples)

    df = pd.DataFrame({"time": t_vals, "y": y_vals})

    model = StaticTAM(formula="y ~ f(time, m=3) + bispec(time, k=4, n_fft=64)")
    model.fit(df)

    preds = model.predict(df)
    assert "Estimatedy" in preds.columns
    assert len(preds) == n_samples
    assert not preds["Estimatedy"].isna().any()

    # Decompose prediction check
    dec = model.decompose_prediction(df)
    assert "effect_f_time" in dec or "f_time" in dec
    assert "effect_bispec_time" in dec or "bispec_time" in dec


def test_bispectrum_tensor_product_interaction():
    """Tests that bispec() functions cleanly inside tensor products te()."""
    n_samples = 150
    t_vals = np.linspace(0, 5, n_samples)
    temp_vals = np.linspace(10, 30, n_samples)
    y_vals = np.sin(t_vals) * (temp_vals / 10.0) + 0.05 * np.random.randn(n_samples)

    df = pd.DataFrame({"time": t_vals, "temp": temp_vals, "y": y_vals})

    model = StaticTAM(formula="y ~ te(bispec(time, k=2, n_fft=32), s(temp, k=5))")
    model.fit(df)

    preds = model.predict(df)
    assert len(preds) == n_samples
    assert not preds["Estimatedy"].isna().any()


def test_bispectrum_different_n_fft_yields_different_phi():
    """Verifies that different n_fft parameter values produce distinct feature maps Phi."""
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    effect_64 = BispectrumEffect("x", k=4, n_fft=64)
    effect_128 = BispectrumEffect("x", k=4, n_fft=128)

    phi_64 = effect_64.build_feature_map(x)
    phi_128 = effect_128.build_feature_map(x)

    # Different n_fft must scale the frequencies differently and produce distinct features
    diff = torch.norm(phi_64 - phi_128).item()
    assert diff > 1e-3, f"Expected distinct feature maps for n_fft=64 vs 128, but diff was {diff}"


def test_bispectrum_factory_parses_window_alias():
    """Verifies bispec(x, window=48) builds BispectrumEffect with FFT length 48."""
    from tam.common.utils import parse_formula_to_terms
    from tam.model.spectrum._factory import create_effects_from_parsed_terms

    _, parsed_win = parse_formula_to_terms("Load ~ bispec(x, k=4, window=48)")
    eff_win = [
        e for e in create_effects_from_parsed_terms(parsed_win, {}, 0.0, include_offset=False)
        if isinstance(e, BispectrumEffect)
    ][0]
    assert eff_win.n_fft == 48

    _, parsed_nfft = parse_formula_to_terms("Load ~ bispec(x, k=4, n_fft=64)")
    eff_nfft = [
        e for e in create_effects_from_parsed_terms(parsed_nfft, {}, 0.0, include_offset=False)
        if isinstance(e, BispectrumEffect)
    ][0]
    assert eff_nfft.n_fft == 64

    _, parsed_conflict = parse_formula_to_terms("Load ~ bispec(x, k=4, n_fft=64, window=48)")
    with pytest.raises(ValueError, match="n_fft.*window|window.*n_fft"):
        create_effects_from_parsed_terms(parsed_conflict, {}, 0.0, include_offset=False)


def _coupled_signal(n, k1=12, k2=18, n_fft=128, phases=(0.4, 1.1), third_phase=None, detune=0.0, seed=0):
    t = torch.arange(n, dtype=torch.float64)
    w1, w2 = 2 * math.pi * k1 / n_fft, 2 * math.pi * k2 / n_fft
    phi3 = phases[0] + phases[1] if third_phase is None else third_phase
    return torch.cos(w1 * t + phases[0]) + torch.cos(w2 * t + phases[1]) + torch.cos((w1 + w2 + detune) * t + phi3)


def test_bispectrum_columns_see_phase_coupling():
    """Phase-locked components give a bispectral product of constant phase; an unlocked third component rotates it."""
    n_fft = 128
    kw = dict(k=1, n_fft=n_fft, triads=[(12, 18)])
    locked = _coupled_signal(6000, n_fft=n_fft)
    eff = BispectrumEffect("x", **kw)
    eff.initialize(locked)
    phi = eff.build_feature_map(locked)[n_fft:]                  # leave the window warm-up
    re, im = phi[:, 0], phi[:, 1]
    assert float(re.mean()) > 0.5 and float(im.mean().abs()) < 0.3 * float(re.mean())
    assert float(re.std()) < 0.2 * float(re.mean())           # a constant product (a steady signal)
    unlocked = _coupled_signal(6000, n_fft=n_fft, detune=2 * math.pi * 0.37 / n_fft, third_phase=0.3)
    eff2 = BispectrumEffect("x", **kw)
    eff2.initialize(unlocked)
    phi2 = eff2.build_feature_map(unlocked)[n_fft:]
    assert float(phi2[:, 0].mean().abs()) < 0.1 * float(phi2[:, 0].std())
    assert float(phi2[:, 1].mean().abs()) < 0.1 * float(phi2[:, 1].std())


def test_bispectrum_causality_and_state_continuation():
    g = torch.Generator().manual_seed(0)
    x = torch.tanh(torch.randn(2, 400, dtype=torch.float64, generator=g))
    eff = BispectrumEffect("x", k=4, n_fft=32)
    eff.initialize(x)
    whole = eff.build_feature_map(x)
    x_mod = x.clone()
    x_mod[:, 250:] += 3.0
    assert torch.equal(eff.build_feature_map(x_mod)[:, :250], whole[:, :250])
    eff.build_feature_map(x[:, :170])
    rest = eff.build_feature_map(x[:, 170:], state_in=eff._state_out)
    assert torch.allclose(rest, whole[:, 170:], atol=1e-9)
    assert eff._state_out["hist"].shape == (2, 31, 1)


def test_bispectrum_columns_are_distinct_per_triad_and_full_rank_on_noise():
    """Four columns per triad, none repeated across triads (the fourth is the cosine of the biphase): rank 4 k on noise with k = 8."""
    eff = BispectrumEffect("x", k=8, n_fft=128)
    sums = [k1 + k2 for k1, k2 in eff.triads]
    assert len(sums) != len(set(sums))                      # triads with the same sum exist: the old third modulus column repeated
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, dtype=torch.float64, generator=g)
    phi = eff.build_feature_map(x)
    assert phi.shape == (2000, 32) and int(torch.linalg.matrix_rank(phi).item()) == 32
    assert torch.isfinite(phi).all() and float(phi[:, 3::4].abs().max()) <= 1.0 + 1e-9      # a cosine


def test_bispectrum_biphase_column_is_one_for_phase_locked_components():
    n_fft = 128
    locked = _coupled_signal(6000, n_fft=n_fft)
    eff = BispectrumEffect("x", k=1, n_fft=n_fft, triads=[(12, 18)])
    eff.initialize(locked)
    cos_phase = eff.build_feature_map(locked)[n_fft:, 3]
    assert float(cos_phase.mean()) > 0.95
    unlocked = _coupled_signal(6000, n_fft=n_fft, detune=2 * math.pi * 0.37 / n_fft, third_phase=0.3)
    eff2 = BispectrumEffect("x", k=1, n_fft=n_fft, triads=[(12, 18)])
    eff2.initialize(unlocked)
    assert abs(float(eff2.build_feature_map(unlocked)[n_fft:, 3].mean())) < 0.2


def test_bispectrum_reads_one_column_and_refuses_isolated_point_extrapolation():
    with pytest.raises(ValueError, match="one column"):
        BispectrumEffect("x", additional_features=["z"])
    with pytest.raises(ValueError, match="temporal effect"):
        BispectrumEffect("x", extrapolate="linear")


def test_bispectrum_reads_its_column_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([BispectrumEffect("x")]) == {"x"}


def _smooth_panel(days=210):
    """Hourly data in 24 groups (hour of the day) with a very smooth covariate: a seasonal sinusoid, a diurnal one and a little noise.
    The bispectral columns of such a series are nearly collinear. The last 10 days are the test chunk."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=24 * days, freq="h")})
    df["tod"] = df["date"].dt.hour
    t = np.arange(len(df)) / 24.0
    df["x"] = 8 + 6 * np.sin(2 * np.pi * t / 365 * 8) + 3 * np.sin(2 * np.pi * (df["tod"] - 9) / 24) + rng.normal(0, 0.5, len(df))
    df["z"] = np.cos(2 * np.pi * t / 7) + rng.normal(0, 0.2, len(df))
    df["is_we"] = (df["date"].dt.dayofweek >= 5).astype(float)
    df["y"] = 50 - 2.0 * df["x"] + 0.05 * df["x"] ** 2 + 3 * df["z"] + 5 * df["is_we"] + rng.normal(0, 1.0, len(df))
    return df.iloc[: 24 * (days - 10)].reset_index(drop=True), df.iloc[24 * (days - 10):].reset_index(drop=True)


def test_formula_default_smoothing_weight_is_the_benchmark_one():
    """`bispec(x)` starts from log10(lambda) = -5 (the columns of a smooth series are nearly collinear); an explicit `ap=` wins."""
    for spec, expected in (("bispec(x)", 1e-5), ("bispec(x, ap=-3)", 1e-3), ("bispec(x, k=4, window=32)", 1e-5)):
        model = StaticTAM(formula=f"y ~ s(x, k=6) + {spec}", group_col="tod", date_col="date")
        effect = [e for e in model.effects_list_ if isinstance(e, BispectrumEffect)][0]
        assert effect.lambda_p == pytest.approx(expected), (spec, effect.lambda_p)


def test_a_window_longer_than_the_groups_warns():
    """No window ever fills when the training groups are shorter than the window: one UserWarning says so."""
    x = torch.randn(3, 40, dtype=torch.float64)
    with pytest.warns(UserWarning, match="longer than the shortest group"):
        BispectrumEffect("x", n_fft=64).initialize(x)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BispectrumEffect("x", n_fft=16).initialize(x)


def test_default_smoothing_does_not_blow_up_out_of_sample_on_a_smooth_series():
    """With the weight the model uses for other terms (log10 = -9) the unpenalised solve of the nearly collinear columns amplifies small differences
    between the training and the test chunk (test RMSE about 90 times the baseline); the default of `bispec` must stay clearly below that."""
    train, test = _smooth_panel()
    col = "Estimatedy"

    def test_rmse(formula):
        model = StaticTAM(formula=formula, group_col="tod", date_col="date").fit(train)
        return float(np.sqrt(np.mean((model.predict(test)[col].to_numpy() - test["y"].to_numpy()) ** 2)))

    default = test_rmse("y ~ s(x, k=6) + l(z) + bispec(x)")
    weak = test_rmse("y ~ s(x, k=6) + l(z) + bispec(x, ap=-9)")
    assert default <= 0.5 * weak, (default, weak)
    # the columns of the test chunk stay in the range of the training ones
    effect = BispectrumEffect("x", n_fft=128)
    pivot = lambda d: torch.tensor(np.stack([g["x"].to_numpy() for _, g in d.groupby("tod")]), dtype=torch.float64)
    xtr, xte = pivot(train), pivot(test)
    effect.initialize(xtr)
    ftr = effect.build_feature_map(xtr)
    fte = effect.build_feature_map(xte, state_in=effect._state_out)
    assert float(fte.abs().max()) <= 6.0 * float(ftr.abs().max())
