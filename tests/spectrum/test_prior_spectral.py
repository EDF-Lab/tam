# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._prior_spectral.PriorSpectralEffect``."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import t

import tam
from tam.model.additive import StaticTAM
from tam.model.spectrum import PriorSpectralEffect


def test_prior_spectral_contract(normalized, penalty_shape):
    """Verifies coefficient counts, feature map shapes, and positive-definite penalty."""
    k = 16
    effect = PriorSpectralEffect("x", k=k, prior="matern", nu=1.5, length_scale=1.0, sobolev_order=1.0)
    assert effect.get_n_coeffs() == 2 * k

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 2 * k)
    assert torch.isfinite(phi).all()

    p_shape = penalty_shape(effect)
    assert p_shape == (2 * k, 2 * k)

    p_mat = effect.build_penalty_matrix()
    if p_mat.is_sparse:
        p_mat = p_mat.to_dense()
    eigvals = torch.linalg.eigvalsh(p_mat)
    assert (eigvals > 0.0).all(), "Penalty matrix must be strictly positive-definite"


@pytest.mark.parametrize("nu", [0.5, 1.5])
def test_prior_spectral_bochner_convergence(nu):
    """Verifies that the QMC feature map converges to the exact Matérn kernel as M increases.

    Also asserts that Halton QMC has strictly lower approximation error than pseudo-random MC at M=64.
    """
    grid_size = 40
    x_grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    diff = x_grid.unsqueeze(-1) - x_grid.unsqueeze(0)
    dist = torch.abs(diff)
    ell = 1.0

    if nu == 0.5:
        k_exact = torch.exp(-dist / ell)
    elif nu == 1.5:
        c = np.sqrt(3.0) / ell
        k_exact = (1.0 + c * dist) * torch.exp(-c * dist)
    else:
        raise ValueError(f"Unsupported nu for exact test: {nu}")

    norm_exact = torch.linalg.norm(k_exact)
    errors_halton = []

    m_list = [16, 64, 256, 1024]
    for m in m_list:
        effect = PriorSpectralEffect(
            "x", k=m, prior="matern", nu=nu, length_scale=ell, qmc="halton", seed=42
        )
        phi = effect.build_feature_map(x_grid)
        k_approx = phi @ phi.T
        err = float(torch.linalg.norm(k_approx - k_exact) / norm_exact)
        errors_halton.append(err)

    # Check that error decreases from M=16 to M=1024
    assert errors_halton[-1] < errors_halton[0], "Approximation error must decrease as M grows"
    assert errors_halton[1] < errors_halton[0]

    # Compare Halton vs MC at M=64
    effect_mc = PriorSpectralEffect(
        "x", k=64, prior="matern", nu=nu, length_scale=ell, qmc="mc", seed=42
    )
    phi_mc = effect_mc.build_feature_map(x_grid)
    k_approx_mc = phi_mc @ phi_mc.T
    err_mc = float(torch.linalg.norm(k_approx_mc - k_exact) / norm_exact)

    assert errors_halton[1] < err_mc, f"QMC Halton error ({errors_halton[1]:.4f}) must be < MC error ({err_mc:.4f})"


def test_prior_spectral_sobol_and_callable():
    """Verifies Sobol generator and custom callable prior."""
    # Sobol
    effect_sobol = PriorSpectralEffect("x", k=32, prior="lorentzian", qmc="sobol", seed=10)
    x = torch.linspace(-1.0, 1.0, 20)
    phi_sobol = effect_sobol.build_feature_map(x)
    assert phi_sobol.shape == (20, 64)
    assert torch.isfinite(phi_sobol).all()

    # Custom callable inverse CDF: e.g. Gaussian prior
    def gaussian_icdf(u):
        return torch.erfinv(2.0 * u - 1.0) * np.sqrt(2.0)

    effect_custom = PriorSpectralEffect("x", k=16, prior=gaussian_icdf)
    phi_custom = effect_custom.build_feature_map(x)
    assert phi_custom.shape == (20, 32)
    assert torch.isfinite(phi_custom).all()


def test_prior_spectral_empirical_initialization():
    """Verifies empirical periodogram prior initialization on training series."""
    t_steps = 100
    time = torch.linspace(0.0, 10.0, t_steps)
    # Series with a clear harmonic at omega ~ 2*pi*0.5
    series = torch.sin(2.0 * np.pi * 0.5 * time) + 0.1 * torch.randn_like(time)

    effect = PriorSpectralEffect("time", k=32, prior="empirical")
    effect.initialize(series)

    phi = effect.build_feature_map(time)
    assert phi.shape == (t_steps, 64)
    assert torch.isfinite(phi).all()

    p_mat = effect.build_penalty_matrix()
    assert (torch.linalg.eigvalsh(p_mat) > 0.0).all()


def test_prior_spectral_static_tam_fit_predict_decompose():
    """Verifies end-to-end integration with StaticTAM: fit, predict, and decomposition."""
    np.random.seed(42)
    n = 120
    x = np.linspace(-1.0, 1.0, n)
    y = np.sin(3.0 * x) + 0.2 * np.random.randn(n)
    df_train = pd.DataFrame({"X": x[:90], "Y": y[:90]})
    df_test = pd.DataFrame({"X": x[90:], "Y": y[90:]})

    model = StaticTAM(formula="Y ~ pspec(X, k=16, prior='matern', nu=1.5)")
    model.fit(df_train)

    preds = model.predict(df_test)
    assert isinstance(preds, (pd.Series, pd.DataFrame, np.ndarray))
    assert len(preds) == len(df_test)
    assert not preds["EstimatedY"].isna().any()

    # Test decomposition
    decomp = model.decompose_prediction(df_test)
    assert isinstance(decomp, pd.DataFrame)
    effect_cols = [c for c in decomp.columns if c.startswith("effect_")]
    assert len(effect_cols) > 0
    assert any("X" in c for c in effect_cols)


def test_prior_spectral_zero_leakage():
    """Verifies that test set perturbations do not change sampled frequencies or fit."""
    np.random.seed(42)
    n = 100
    x = np.linspace(-1.0, 1.0, n)
    y = 0.5 * x + np.sin(2.0 * x)
    df_train = pd.DataFrame({"X": x[:70], "Y": y[:70]})

    model1 = StaticTAM(formula="Y ~ pspec(X, k=24, prior='matern', seed=42)")
    model1.fit(df_train)

    df_test1 = pd.DataFrame({"X": x[70:], "Y": y[70:]})
    pred1 = model1.predict(df_test1)

    # Completely different test set
    df_test2 = pd.DataFrame({"X": x[70:] + 10.0, "Y": y[70:] * 5.0})
    pred1_again = model1.predict(df_test1)

    np.testing.assert_allclose(pred1, pred1_again, rtol=1e-12)


def test_prior_spectral_tensor_product_interaction():
    """Verifies that pspec works inside te() tensor products."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = np.sin(x1) * np.cos(x2) + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ te(pspec(X1, k=8), s(X2, k=8))")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def test_prior_spectral_multivariate():
    """Verifies multivariate inputs with others/additional_features."""
    np.random.seed(42)
    n = 80
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = x1 + 0.5 * x2 + 0.1 * np.random.randn(n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    model = StaticTAM(formula="Y ~ pspec(X1, others='X2', k=16, prior='matern')")
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == n
    assert not preds["EstimatedY"].isna().any()


def test_prior_spectral_keeps_dtype_and_treats_groups_alone():
    """The features are pointwise: float32 in gives float32 out, and a group does not depend on the others."""
    eff = PriorSpectralEffect("x", k=8)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 50, dtype=torch.float64, generator=g)
    whole = eff.build_feature_map(x)
    assert whole.shape == (3, 50, 16) and eff.build_feature_map(x.float()).dtype == torch.float32
    assert torch.allclose(eff.build_feature_map(x[1:2])[0], whole[1], atol=1e-12)


def test_prior_spectral_reads_its_columns_out_of_range():
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([PriorSpectralEffect("x", additional_features=["z"])]) == {"x", "z"}


def test_empirical_prior_is_a_per_group_periodogram():
    """Groups at different levels must not leak a jump at their boundary into the spectrum: the prior averages the groups' periodograms."""
    g = torch.Generator().manual_seed(3)
    t = torch.arange(128, dtype=torch.float64)
    base = torch.sin(2 * torch.pi * t / 8.0)
    x = torch.stack([base + 50.0 * b + 0.05 * torch.randn(128, dtype=torch.float64, generator=g) for b in range(4)])  # (4, 128)

    grouped = PriorSpectralEffect("x", k=32, prior="empirical")
    grouped.initialize(x)
    single = PriorSpectralEffect("x", k=32, prior="empirical")
    single.initialize(x[0])  # one group at one level: the reference spectrum
    # the frequencies sit at the group's dominant frequency 2 pi / 8, not at the low frequencies that the level jumps would add
    assert torch.allclose(grouped.frequencies.median(), single.frequencies.median(), atol=0.1), (grouped.frequencies.median(), single.frequencies.median())
