# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Tests for ``dist_kwargs={"target_transform": ...}`` of the distributional :class:`StaticTAM`: ``"asinh"`` for a skewed real-valued target
(negative values allowed) and ``"logit"`` for a target bounded in an interval. ``log_target`` keeps its meaning and its default.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tam import StaticTAM

_FORMULAS = {"mu": "y ~ s(x)", "sigma": "y ~ s(x)"}
_LOSS = {"mu": "l2", "sigma": "gamma"}
_TAUS = (0.05, 0.5, 0.95)


def _model(**dist_kwargs):
    return StaticTAM(_FORMULAS, loss=_LOSS, dist_kwargs=dist_kwargs)


def _johnson_su(rng, n):
    """A skewed, heavy-tailed, real-valued target with negatives: y = -1 + 4 sinh(mu(x) + 1.2 z), the asinh-normal (Johnson SU) family."""
    x = rng.uniform(0.0, 1.0, n)
    y = -1.0 + 4.0 * np.sinh(1.5 * (x - 0.5) + 1.2 * rng.standard_normal(n))
    return pd.DataFrame({"x": x, "y": y})


def _bounded(rng, n):
    x = rng.uniform(0.0, 1.0, n)
    mean = 0.15 + 0.7 * x
    y = rng.beta(mean * 8.0, (1.0 - mean) * 8.0)
    return pd.DataFrame({"x": x, "y": y})


def _coverage(frame, quantiles, tau):
    return float(np.mean(frame["y"].to_numpy() <= quantiles[f"q{tau}"].to_numpy()))


def test_asinh_recovers_the_tails_of_a_skewed_target_with_negatives():
    rng = np.random.default_rng(0)
    train, test = _johnson_su(rng, 6000), _johnson_su(rng, 6000)
    assert (train["y"] < 0).any()
    asinh = _model(target_transform="asinh", tail_family="normal").fit(train)
    plain = _model(log_target=False, tail_family="normal").fit(train)
    quantiles = asinh.predict_quantiles(test, _TAUS)
    assert abs(_coverage(test, quantiles, 0.05) - 0.05) < 0.02
    assert abs(_coverage(test, quantiles, 0.95) - 0.95) < 0.02
    # a symmetric law on the response scale misplaces the tails of a skewed target by more than two points
    plain_quantiles = plain.predict_quantiles(test, _TAUS)
    plain_errors = [abs(_coverage(test, plain_quantiles, tau) - tau) for tau in (0.05, 0.95)]
    assert max(plain_errors) > 0.02


def test_asinh_quantiles_do_not_cross_and_the_median_is_the_back_transformed_location():
    rng = np.random.default_rng(1)
    train = _johnson_su(rng, 3000)
    model = _model(target_transform="asinh").fit(train)
    quantiles = model.predict_quantiles(train, (0.1, 0.5, 0.9))
    assert (quantiles["q0.1"] <= quantiles["q0.5"]).all() and (quantiles["q0.5"] <= quantiles["q0.9"]).all()
    np.testing.assert_allclose(model.predict_median(train), quantiles["q0.5"].to_numpy(), rtol=1e-8, atol=1e-8)


def test_asinh_cdf_and_crps_and_anomaly_score_work_through_the_transform():
    rng = np.random.default_rng(2)
    train = _johnson_su(rng, 3000)
    model = _model(target_transform="asinh", tail_family="normal").fit(train)
    pit = model.cdf(train)
    assert 0.0 <= pit.min() and pit.max() <= 1.0
    assert abs(float(np.mean(pit)) - 0.5) < 0.03
    assert np.isfinite(model.crps(train)).all() and (model.crps(train) >= 0).all()
    score = model.anomaly_score(train)
    np.testing.assert_allclose(score["predicted_median"].to_numpy(), model.predict_median(train))


def test_logit_quantiles_stay_inside_the_bounds_and_cover():
    rng = np.random.default_rng(3)
    train, test = _bounded(rng, 5000), _bounded(rng, 5000)
    model = _model(target_transform="logit", bounds=(0.0, 1.0), tail_family="normal").fit(train)
    quantiles = model.predict_quantiles(test, (0.01, 0.05, 0.5, 0.95, 0.99))
    assert quantiles.to_numpy().min() > 0.0 and quantiles.to_numpy().max() < 1.0
    assert abs(_coverage(test, quantiles, 0.05) - 0.05) < 0.03
    assert abs(_coverage(test, quantiles, 0.95) - 0.95) < 0.03
    # a normal law on the response scale puts quantiles outside the bounds
    plain = _model(log_target=False, tail_family="normal").fit(train).predict_quantiles(test, (0.01, 0.99))
    assert plain.to_numpy().min() < 0.0 or plain.to_numpy().max() > 1.0


def test_logit_estimates_and_stores_bounds_when_none_are_given():
    rng = np.random.default_rng(4)
    train = _bounded(rng, 3000)
    model = _model(target_transform="logit", tail_family="normal").fit(train)
    low, high = model._transform_params_["bounds"]
    assert low < train["y"].min() and high > train["y"].max()
    quantiles = model.predict_quantiles(train, (0.001, 0.999))
    assert quantiles.to_numpy().min() > low and quantiles.to_numpy().max() < high


def test_logit_values_at_the_bounds_are_nudged_not_rejected():
    rng = np.random.default_rng(5)
    train = _bounded(rng, 2000)
    train.loc[:20, "y"] = 0.0
    train.loc[21:40, "y"] = 1.0
    model = _model(target_transform="logit", bounds=(0.0, 1.0)).fit(train)
    assert np.isfinite(model.predict_median(train)).all()


def test_logit_rejects_values_outside_the_given_bounds():
    rng = np.random.default_rng(6)
    train = _bounded(rng, 500)
    train.loc[0, "y"] = 1.5
    with pytest.raises(ValueError, match=r"outside the bounds"):
        _model(target_transform="logit", bounds=(0.0, 1.0)).fit(train)


def test_an_unknown_transform_or_a_contradiction_with_log_target_is_rejected():
    with pytest.raises(ValueError, match="target_transform"):
        _model(target_transform="sqrt")
    with pytest.raises(ValueError, match="log_target"):
        _model(target_transform="asinh", log_target=True)
    with pytest.raises(ValueError, match="bounds"):
        _model(target_transform="logit", bounds=(1.0, 0.0))


def test_a_mixture_does_not_accept_a_target_transform():
    with pytest.raises(ValueError, match="mixture"):
        StaticTAM("y ~ s(x)", mixture_components=2, dist_kwargs={"target_transform": "asinh"})


def test_the_default_and_log_target_true_are_the_same_log_fit():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 2000)
    train = pd.DataFrame({"x": x, "y": np.exp(1.0 + x + 0.3 * rng.standard_normal(2000))})
    default = _model().fit(train).predict_quantiles(train, _TAUS)
    explicit = _model(log_target=True).fit(train).predict_quantiles(train, _TAUS)
    named = _model(target_transform="log").fit(train).predict_quantiles(train, _TAUS)
    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(default, named)
    assert default.to_numpy().min() > 0.0


def test_the_conformal_intervals_cover_a_target_modelled_through_asinh():
    rng = np.random.default_rng(8)
    train, calibration, test = (_johnson_su(rng, 3000) for _ in range(3))
    model = _model(target_transform="asinh", tail_family="normal").fit(train)
    model.calibrate_conformal(calibration, alpha=0.1, studentized=True)
    intervals = model.predict_intervals(test)
    inside = (test["y"].to_numpy() >= intervals["Lower"].to_numpy()) & (test["y"].to_numpy() <= intervals["Upper"].to_numpy())
    assert abs(float(inside.mean()) - 0.9) < 0.03


def test_the_back_transform_inverts_the_transform():
    rng = np.random.default_rng(9)
    train = _johnson_su(rng, 2000)
    for kwargs in ({"target_transform": "asinh"}, {"log_target": False}):
        model = _model(**kwargs).fit(train)
        back = model._from_model_scale(model._to_model_scale(train["y"].to_numpy()), "round trip")
        np.testing.assert_allclose(back, train["y"].to_numpy(), rtol=1e-8, atol=1e-8)
    bounded = _bounded(rng, 2000)
    logit = _model(target_transform="logit", bounds=(0.0, 1.0)).fit(bounded)
    back = logit._from_model_scale(logit._to_model_scale(bounded["y"].to_numpy()), "round trip")
    np.testing.assert_allclose(back, bounded["y"].to_numpy(), rtol=1e-8, atol=1e-8)
