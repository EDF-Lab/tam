# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``target_transform`` of a plain :class:`StaticTAM` (``log``, ``log1p``, ``asinh``, ``logit``, ``boxcox``): the model is fitted on
``g(y)``, ``predict`` returns the response scale, ``predict_quantiles`` inverts the quantiles of the model-scale residual law (exact
for a monotone ``g``). The same transforms serve the distributional mode through ``dist_kwargs``.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.statistics.estimation import _target_transform as tt

NAMES = ["log", "log1p", "asinh", "logit", "boxcox"]


def _positive(n=4000, seed=0, sigma=0.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "y": np.exp(1.0 + 1.5 * x + sigma * rng.standard_normal(n))})


def _bounded(n=4000, seed=1, low=0.0, high=100.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = -1.0 + 3.0 * x + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": low + (high - low) / (1 + np.exp(-z))})


def _real(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "y": -1.0 + 4.0 * np.sinh(1.5 * (x - 0.5) + 0.8 * rng.standard_normal(n))})


def _frame_for(name):
    return {"logit": _bounded(), "asinh": _real()}.get(name, _positive())


@pytest.mark.parametrize("name", NAMES)
def test_forward_then_inverse_is_the_identity(name):
    y = _frame_for(name)["y"].to_numpy()
    kwargs = {"bounds": (0.0, 100.0)} if name == "logit" else {}
    params = tt.estimate_params(name, y, kwargs)
    z = tt.forward(name, params, y)
    assert np.all(np.isfinite(z))
    assert np.allclose(tt.inverse(name, params, z), y, rtol=1e-8, atol=1e-8)
    order = np.argsort(y)
    assert np.all(np.diff(z[order]) >= -1e-12)  # monotone increasing


@pytest.mark.parametrize("name", ["log", "log1p", "boxcox"])
def test_domain_is_checked(name):
    y = np.array([1.0, 2.0, -5.0, 3.0])
    with pytest.raises(ValueError, match="values"):
        tt.estimate_params(name, y, {})


def test_unknown_transform_and_bad_kwargs_raise():
    with pytest.raises(ValueError):
        ta.StaticTAM("y ~ s(x)", target_transform="sqrt")
    with pytest.raises(ValueError):
        ta.StaticTAM("y ~ s(x)", target_transform="log", transform_kwargs={"bounds": (0, 1)})
    with pytest.raises(ValueError):
        ta.StaticTAM("y ~ s(x)", loss="poisson", target_transform="log")
    with pytest.raises(ValueError):
        ta.StaticTAM({"mu": "y ~ s(x)"}, target_transform="log")


def test_no_transform_is_the_previous_behaviour():
    frame = _positive()
    plain = ta.StaticTAM("y ~ s(x, k=8)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    explicit = ta.StaticTAM("y ~ s(x, k=8)", target_transform=None).fit(frame).predict(frame)["Estimatedy"].to_numpy()
    assert np.array_equal(plain, explicit)


@pytest.mark.parametrize("name", NAMES)
def test_predict_is_the_inverse_of_the_fit_on_the_transformed_target(name):
    frame = _frame_for(name)
    kwargs = {"bounds": (0.0, 100.0)} if name == "logit" else {}
    model = ta.StaticTAM("y ~ s(x, k=8)", target_transform=name, transform_kwargs=kwargs).fit(frame)
    params = tt.estimate_params(name, frame["y"].to_numpy(), kwargs)
    manual_frame = frame.assign(y=tt.forward(name, params, frame["y"].to_numpy()))
    manual = ta.StaticTAM("y ~ s(x, k=8)").fit(manual_frame).predict(frame)["Estimatedy"].to_numpy()
    expected = tt.inverse(name, params, manual)
    assert np.allclose(model.predict(frame)["Estimatedy"].to_numpy(), expected, rtol=1e-8, atol=1e-8)


def test_log_predictions_are_positive_and_beat_the_raw_fit_on_a_multiplicative_target():
    train, test = _positive(seed=3, sigma=0.7), _positive(n=6000, seed=4, sigma=0.7)
    log_model = ta.StaticTAM("y ~ s(x, k=8)", target_transform="log").fit(train)
    pred = log_model.predict(test)["Estimatedy"].to_numpy()
    assert np.all(pred > 0)
    truth = np.exp(1.0 + 1.5 * test["x"].to_numpy())  # the median curve
    raw = ta.StaticTAM("y ~ s(x, k=8)").fit(train).predict(test)["Estimatedy"].to_numpy()
    assert np.mean(np.abs(np.log(pred / truth))) < np.mean(np.abs(np.log(np.clip(raw, 1e-3, None) / truth)))


def test_mean_correction_removes_the_jensen_bias_of_the_log():
    train, test = _positive(n=20000, seed=5, sigma=0.8), _positive(n=20000, seed=6, sigma=0.8)
    plain = ta.StaticTAM("y ~ s(x, k=8)", target_transform="log").fit(train)
    corrected = ta.StaticTAM("y ~ s(x, k=8)", target_transform="log", transform_kwargs={"mean_correction": True}).fit(train)
    target_mean = test["y"].mean()
    bias_plain = abs(plain.predict(test)["Estimatedy"].mean() / target_mean - 1.0)
    bias_corrected = abs(corrected.predict(test)["Estimatedy"].mean() / target_mean - 1.0)
    assert bias_plain > 0.15 and bias_corrected < 0.03


@pytest.mark.parametrize("name", NAMES)
def test_quantiles_are_ordered_and_hold_their_coverage(name):
    train, test = _frame_for(name), {"logit": _bounded(seed=11), "asinh": _real(seed=12)}.get(name, _positive(seed=13))
    kwargs = {"bounds": (0.0, 100.0)} if name == "logit" else {}
    model = ta.StaticTAM("y ~ s(x, k=8)", target_transform=name, transform_kwargs=kwargs).fit(train)
    taus = (0.1, 0.5, 0.9)
    q = model.predict_quantiles(test, taus)
    assert list(q.columns) == ["q0.1", "q0.5", "q0.9"]
    assert np.all(q["q0.1"].to_numpy() <= q["q0.5"].to_numpy()) and np.all(q["q0.5"].to_numpy() <= q["q0.9"].to_numpy())
    for tau in taus:
        assert abs(np.mean(test["y"].to_numpy() <= q[f"q{tau}"].to_numpy()) - tau) < 0.04


def test_quantiles_are_the_inverse_of_the_model_scale_quantiles():
    frame = _positive()
    model = ta.StaticTAM("y ~ s(x, k=8)", target_transform="log").fit(frame)
    q = model.predict_quantiles(frame, (0.25, 0.75))
    median = model.predict_quantiles(frame, (0.5,))["q0.5"].to_numpy()
    # g^{-1} is monotone: the quantile of y is g^{-1} of the quantile of g(y); the median of the residual law is ~0
    assert np.allclose(median, model.predict(frame)["Estimatedy"].to_numpy(), rtol=0.05)
    assert np.all(q["q0.25"].to_numpy() < q["q0.75"].to_numpy())


def test_logit_quantiles_stay_inside_the_bounds_whatever_the_features():
    frame = _bounded()
    model = ta.StaticTAM("y ~ s(x, k=8)", target_transform="logit", transform_kwargs={"bounds": (0.0, 100.0)}).fit(frame)
    shocked = pd.DataFrame({"x": np.linspace(-3, 4, 50)})
    q = model.predict_quantiles(shocked, (0.01, 0.5, 0.99))
    assert np.all(q.to_numpy() >= 0.0) and np.all(q.to_numpy() <= 100.0)
    assert np.all(model.predict(shocked)["Estimatedy"].to_numpy() <= 100.0)


def test_logit_rejects_values_outside_the_given_bounds():
    frame = _bounded()
    with pytest.raises(ValueError, match="bounds"):
        ta.StaticTAM("y ~ s(x)", target_transform="logit", transform_kwargs={"bounds": (10.0, 90.0)}).fit(frame)


@pytest.mark.parametrize("route", ["auto_fit", "grid_search_fit"])
def test_the_other_fitting_routes_use_the_transform(route):
    frame = _positive(n=1500)
    if route == "auto_fit":
        model = ta.StaticTAM("y ~ s(x, k=10)", target_transform="log").auto_fit(frame)
    else:
        model = ta.StaticTAM("y ~ s(x, k=10, ap='pen')", target_transform="log").grid_search_fit(frame, frame, {"pen": [-6.0, -2.0]})
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.all(pred > 0) and np.median(np.abs(np.log(pred / frame["y"].to_numpy()))) < 1.0
    assert model.predict_quantiles(frame, (0.1, 0.9)).shape == (len(frame), 2)


@pytest.mark.parametrize("name", ["log1p", "boxcox"])
def test_the_distributional_mode_accepts_the_new_transforms(name):
    frame = _positive(n=3000)
    model = ta.StaticTAM({"mu": "y ~ s(x)", "sigma": "y ~ s(x)"}, loss={"mu": "l2", "sigma": "gamma"},
                         dist_kwargs={"target_transform": name}).fit(frame)
    q = model.predict_quantiles(frame, (0.1, 0.9))
    assert np.all(q["q0.1"].to_numpy() < q["q0.9"].to_numpy()) and np.all(q["q0.1"].to_numpy() > 0)
