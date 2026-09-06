# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Tests for the distributional mode of :class:`StaticTAM`, the log-location-scale fit produced by a
``{param: formula}`` dict with ``loss={"mu": "l2", "sigma": "gamma"}``.

Covers median & heteroscedastic-scale recovery, non-crossing quantiles, coverage calibration, the
probability integral transform, anomaly scoring, CRPS (closed form vs quantile integral), and the
automatic Normal/Student-t tail diagnostic.
"""
import numpy as np
import pandas as pd
from scipy import stats

from tam import StaticTAM

_FORMULAS = {"mu": "y ~ s(x)", "sigma": "y ~ s(x)"}
_LOSS = {"mu": "l2", "sigma": "gamma"}


def _distributional(**kwargs):
    return StaticTAM(_FORMULAS, loss=_LOSS, **kwargs)


def _heteroscedastic_lognormal(rng, n, error="normal"):
    x = rng.uniform(0.0, 1.0, n)
    mu = 1.0 + 1.5 * x
    sigma = 0.2 + 0.6 * x
    if error == "student":
        noise = rng.standard_t(3.0, n) / np.sqrt(3.0)  # standardized to unit variance
    else:
        noise = rng.standard_normal(n)
    frame = pd.DataFrame({"x": x, "y": np.exp(mu + sigma * noise)})
    return frame, mu, sigma


def _fit(rng, n=4000, **kwargs):
    train, _, _ = _heteroscedastic_lognormal(rng, n)
    return _distributional(**kwargs).fit(train)


def test_dict_formula_builds_distributional_static_tam():
    model = _distributional(dist_kwargs={"tail_family": "normal"})
    assert isinstance(model, StaticTAM)
    assert model._mode_ == "distributional"


def test_median_and_scale_recovery():
    rng = np.random.default_rng(0)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, mu_te, sigma_te = _heteroscedastic_lognormal(rng, 4000)
    median = model.predict_median(test)
    assert np.corrcoef(median, np.exp(mu_te))[0, 1] > 0.9
    recovered_sigma = np.log(model.predict_quantile(test, 0.8413)) - np.log(model.predict_quantile(test, 0.5))
    assert np.corrcoef(recovered_sigma, sigma_te)[0, 1] > 0.7


def test_quantiles_do_not_cross():
    rng = np.random.default_rng(1)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 4000)
    q = model.predict_quantile(test, [0.05, 0.5, 0.95])
    assert np.all(q[:, 0] < q[:, 1]) and np.all(q[:, 1] < q[:, 2])


def test_coverage_is_calibrated():
    rng = np.random.default_rng(2)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 6000)
    for tau in (0.05, 0.5, 0.95):
        empirical = float(np.mean(test["y"].to_numpy() <= model.predict_quantile(test, tau)))
        assert abs(empirical - tau) < 0.03


def test_probability_integral_transform_is_uniform():
    rng = np.random.default_rng(3)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 5000)
    assert stats.kstest(model.cdf(test), "uniform").statistic < 0.03


def test_anomaly_detection_and_side():
    rng = np.random.default_rng(4)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 5000)
    clean_threshold = np.quantile(model.anomaly_score(test)["anomaly_score"].to_numpy(), 0.99)

    anomalies = test.iloc[:80].copy().reset_index(drop=True)
    anomalies.loc[:39, "y"] *= 12.0     # grossly over-priced
    anomalies.loc[40:, "y"] /= 12.0     # grossly under-priced
    scored = model.anomaly_score(anomalies)
    assert (scored["anomaly_score"].to_numpy() > clean_threshold).mean() > 0.85
    assert (scored.loc[:39, "side"] == "over").all()
    assert (scored.loc[40:, "side"] == "under").all()


def test_predict_quantiles_frame_is_labelled_and_non_crossing():
    rng = np.random.default_rng(11)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 2000)
    frame = model.predict_quantiles(test, taus=(0.1, 0.5, 0.9))
    assert list(frame.columns) == ["q0.1", "q0.5", "q0.9"]
    assert np.all(frame["q0.1"].to_numpy() < frame["q0.5"].to_numpy())
    assert np.all(frame["q0.5"].to_numpy() < frame["q0.9"].to_numpy())


def test_crps_closed_form_matches_quantile_integral():
    rng = np.random.default_rng(5)
    model = _fit(rng, n=3000, dist_kwargs={"tail_family": "normal"})
    train, _, _ = _heteroscedastic_lognormal(rng, 3000)
    closed_form = model.crps(train)
    observed = np.log(train["y"].to_numpy())
    nodes, weights = np.polynomial.legendre.leggauss(64)
    tau_grid, tau_weights = 0.5 * (nodes + 1.0), 0.5 * weights
    integral = np.zeros_like(observed)
    for tau, weight in zip(tau_grid, tau_weights):
        quantile = np.log(model.predict_quantile(train, float(tau)))
        residual = observed - quantile
        integral += weight * 2.0 * (residual * (tau - (residual < 0)))
    assert np.mean(np.abs(closed_form - integral)) / np.mean(closed_form) < 1e-3
    assert np.all(closed_form >= -1e-9)


def test_crps_discriminates_worse_fits():
    rng = np.random.default_rng(6)
    model = _fit(rng, n=3000, dist_kwargs={"tail_family": "normal"})
    train, _, _ = _heteroscedastic_lognormal(rng, 3000)
    corrupted = train.copy()
    corrupted["y"] = corrupted["y"] * 5.0
    assert model.crps(train).mean() < model.crps(corrupted).mean()


def test_auto_tail_diagnostic():
    rng = np.random.default_rng(7)
    normal_train, _, _ = _heteroscedastic_lognormal(rng, 5000, error="normal")
    student_train, _, _ = _heteroscedastic_lognormal(rng, 5000, error="student")
    normal_model = _distributional(dist_kwargs={"tail_family": "auto"}).fit(normal_train)
    student_model = _distributional(dist_kwargs={"tail_family": "auto"}).fit(student_train)
    assert normal_model.tail_family_ == "normal"
    assert student_model.tail_family_ == "student_t"
    assert student_model.nu_ is not None and student_model.nu_ < 20.0


def test_scale_shrinkage_out_of_range_raises():
    for invalid in (-0.1, 1.5):
        try:
            StaticTAM({"mu": "y ~ s(x)"}, dist_kwargs={"scale_shrinkage": invalid})
        except ValueError:
            continue
        raise AssertionError(f"scale_shrinkage={invalid} should have raised")


def test_scale_shrinkage_zero_is_identical():
    rng = np.random.default_rng(8)
    train, _, _ = _heteroscedastic_lognormal(rng, 3000)
    test, _, _ = _heteroscedastic_lognormal(rng, 3000)
    default = _distributional(dist_kwargs={"tail_family": "normal"}).fit(train)
    shrunk_zero = _distributional(dist_kwargs={"tail_family": "normal", "scale_shrinkage": 0.0}).fit(train)
    np.testing.assert_array_equal(
        default.predict_quantile(test, [0.05, 0.5, 0.95]),
        shrunk_zero.predict_quantile(test, [0.05, 0.5, 0.95]),
    )


def test_scale_shrinkage_one_collapses_to_global_constant_scale():
    rng = np.random.default_rng(9)
    train, _, _ = _heteroscedastic_lognormal(rng, 4000)
    test, _, _ = _heteroscedastic_lognormal(rng, 4000)
    model = _distributional(dist_kwargs={"tail_family": "normal", "scale_shrinkage": 1.0}).fit(train)
    _, sigma_hat = model._mu_sigma(test)
    assert np.allclose(sigma_hat, sigma_hat[0])
    assert np.isclose(sigma_hat[0], np.sqrt(model._global_scale_variance_))


def test_scale_shrinkage_monotonically_shrinks_scale_dispersion():
    rng = np.random.default_rng(10)
    train, _, _ = _heteroscedastic_lognormal(rng, 4000)
    test, _, _ = _heteroscedastic_lognormal(rng, 4000)
    dispersions = []
    for lam in (0.0, 0.3, 0.6, 1.0):
        model = _distributional(dist_kwargs={"tail_family": "normal", "scale_shrinkage": lam}).fit(train)
        _, sigma_hat = model._mu_sigma(test)
        dispersions.append(float(np.std(sigma_hat)))
    assert all(later <= earlier + 1e-9 for earlier, later in zip(dispersions, dispersions[1:]))
    assert dispersions[-1] < dispersions[0]
