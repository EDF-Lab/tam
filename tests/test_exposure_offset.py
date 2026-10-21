# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``offset(col)`` in a formula: the raw, unnormalised column enters the linear predictor with a fixed coefficient 1 and no parameter,
``eta = offset + Phi theta``. It is exact for the Gaussian fit, the reweighted fit of the GLM losses, ``auto_fit``,
``grid_search_fit``, ``predict`` and ``decompose_prediction``.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta


def _gaussian(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x_off": rng.normal(10.0, 5.0, n), "x1": rng.uniform(-1, 1, n)}).assign(y=lambda d: d["x_off"] + 2.0 * d["x1"])


def _counts(loss, n=6000, seed=1, slope=0.5):
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(1.0, 50.0, n)
    x1 = rng.uniform(-1, 1, n)
    mean = exposure * np.exp(0.3 + slope * x1)
    if loss == "poisson":
        y = rng.poisson(mean)
    elif loss == "negative_binomial":
        y = rng.negative_binomial(5, 5 / (5 + mean))
    else:  # tweedie
        y = np.where(rng.random(n) < 0.15, 0.0, rng.gamma(2.0, mean / 2.0))
    return pd.DataFrame({"log_E": np.log(exposure), "x1": x1, "y": y.astype(float)})


def test_gaussian_offset_has_a_fixed_coefficient_of_one():
    frame = _gaussian()
    model = ta.StaticTAM("y ~ offset(x_off) + l(x1, ap=2.0)").fit(frame)  # the penalty on l(x1) is huge
    shifted = frame.assign(x_off=frame["x_off"] + 7.5)
    gap = model.predict(shifted)["Estimatedy"].to_numpy() - model.predict(frame)["Estimatedy"].to_numpy()
    assert np.allclose(gap, 7.5, atol=1e-9)
    parts = model.decompose_prediction(frame)
    assert np.allclose(parts["effect_offset_x_off"].to_numpy(), frame["x_off"].to_numpy(), atol=1e-12)


def test_the_offset_adds_no_coefficient():
    frame = _gaussian()
    with_offset = ta.StaticTAM("y ~ offset(x_off) + l(x1)").fit(frame)
    without = ta.StaticTAM("y ~ l(x1)").fit(frame)
    assert with_offset.coefficients_.shape == without.coefficients_.shape


def test_gaussian_fit_recovers_the_rest_of_the_signal():
    frame = _gaussian()
    model = ta.StaticTAM("y ~ offset(x_off) + l(x1)").fit(frame)
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"].to_numpy() - frame["y"].to_numpy()) ** 2)) < 1e-6


def test_two_offsets_add_up():
    frame = _gaussian().assign(x_off2=lambda d: 0.5 * d["x1"] + 3.0)
    frame["y"] = frame["x_off"] + frame["x_off2"] + 2.0 * frame["x1"]
    model = ta.StaticTAM("y ~ offset(x_off) + offset(x_off2) + l(x1)").fit(frame)
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.sqrt(np.mean((pred - frame["y"].to_numpy()) ** 2)) < 1e-6


@pytest.mark.parametrize("loss", ["poisson", "negative_binomial", "tweedie"])
def test_glm_offset_recovers_the_slope_and_scales_with_the_exposure(loss):
    frame = _counts(loss)
    model = ta.StaticTAM("y ~ offset(log_E) + l(x1)", loss=loss).fit(frame)
    low = pd.DataFrame({"log_E": [2.0, 2.0], "x1": [-0.5, 0.5], "y": 0.0})
    pred = model.predict(low)["Estimatedy"].to_numpy()
    assert np.log(pred[1] / pred[0]) == pytest.approx(0.5, abs=0.08)  # slope 0.5 per unit of x1
    doubled = model.predict(low.assign(log_E=low["log_E"] + np.log(2.0)))["Estimatedy"].to_numpy()
    assert np.allclose(doubled, 2.0 * pred, rtol=1e-9)
    parts = model.decompose_prediction(frame)
    assert np.allclose(parts["effect_offset_log_E"].to_numpy(), frame["log_E"].to_numpy(), atol=1e-12)


def test_the_offset_beats_a_free_slope_when_the_exposure_is_collinear_with_the_feature():
    rng = np.random.default_rng(4)
    n = 4000
    x1 = rng.uniform(-1, 1, n)
    log_e = 3.0 + 0.8 * x1 + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "log_E": log_e, "y": rng.poisson(np.exp(log_e + 0.2 - 0.6 * x1)).astype(float)})
    exact = ta.StaticTAM("y ~ offset(log_E) + l(x1)", loss="poisson").fit(frame)
    free = ta.StaticTAM("y ~ l(log_E, ap=1) + l(x1)", loss="poisson").fit(frame)
    test = pd.DataFrame({"x1": rng.uniform(-1, 1, n)})
    test["log_E"] = 3.0 + 0.8 * test["x1"] + 1.0 * rng.normal(size=n)  # the effort varies more on the test stations
    test["y"] = rng.poisson(np.exp(test["log_E"] + 0.2 - 0.6 * test["x1"])).astype(float)

    def deviance(model):
        mu = np.clip(model.predict(test)["Estimatedy"].to_numpy(), 1e-9, None)
        y = test["y"].to_numpy()
        return float(np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))))

    assert deviance(exact) < deviance(free)


@pytest.mark.parametrize("loss", ["l2", "poisson"])
def test_auto_fit_keeps_the_offset(loss):
    frame = _gaussian() if loss == "l2" else _counts("poisson")
    column = "x_off" if loss == "l2" else "log_E"
    model = ta.StaticTAM(f"y ~ offset({column}) + s(x1, k=8)", loss=loss).auto_fit(frame)
    shifted = frame.assign(**{column: frame[column] + 1.0})
    gap = model.predict(shifted)["Estimatedy"].to_numpy() / (1.0 if loss == "l2" else model.predict(frame)["Estimatedy"].to_numpy())
    if loss == "l2":
        assert np.allclose(model.predict(shifted)["Estimatedy"].to_numpy() - model.predict(frame)["Estimatedy"].to_numpy(), 1.0, atol=1e-9)
    else:
        assert np.allclose(gap, np.e, rtol=1e-9)


def test_grid_search_fit_keeps_the_offset():
    frame = _counts("poisson", n=2000)
    train, val = frame.iloc[:1200], frame.iloc[1200:]
    model = ta.StaticTAM("y ~ offset(log_E) + s(x1, k=8, ap='pen')", loss="poisson").grid_search_fit(train, val, {"pen": [-6.0, -2.0]})
    low = pd.DataFrame({"log_E": [2.0], "x1": [0.0], "y": 0.0})
    ratio = model.predict(low.assign(log_E=3.0))["Estimatedy"].iloc[0] / model.predict(low)["Estimatedy"].iloc[0]
    assert ratio == pytest.approx(np.e, rel=1e-9)


def test_a_missing_offset_column_is_reported():
    frame = _gaussian()
    model = ta.StaticTAM("y ~ offset(x_off) + l(x1)").fit(frame)
    with pytest.raises(Exception, match="x_off"):
        model.predict(frame.drop(columns="x_off"))


def test_the_other_model_paths_without_an_offset_are_unchanged():
    frame = _gaussian()
    a = ta.StaticTAM("y ~ l(x1)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    b = ta.StaticTAM("y ~ l(x1)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    assert np.array_equal(a, b)


def test_adaptive_and_kalman_formulas_refuse_an_offset():
    with pytest.raises(ValueError, match="offset"):
        ta.AdaptiveTAM("ResidualY ~ offset(e) + l(x)", update_interval_periods=1, training_window_periods=10, steps_per_period=1)
    with pytest.raises(ValueError, match="offset"):
        ta.KalmanTAM("y ~ offset(e) + l(x)")


def test_a_mixture_refuses_an_offset():
    with pytest.raises(ValueError, match="offset"):
        ta.StaticTAM("y ~ offset(e) + l(x)", mixture_components=2)


def test_another_term_may_not_read_the_offset_column():
    with pytest.raises(ValueError, match="own column"):
        ta.StaticTAM("y ~ offset(e) + l(e)")


def test_summary_lists_the_offset_with_no_parameter():
    model = ta.StaticTAM("y ~ offset(x_off) + l(x1)").fit(_gaussian())
    table = model.summary()
    row = table[table["Type"] == "ExposureOffset"].iloc[0]
    assert row["Feature"] == "x_off" and row["Complexity (D)"] == 0


def test_the_bootstrap_does_not_project_the_offset_on_the_design():
    from tam.model.diagnostics import compute_bootstrap_significance
    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame({"x_off": rng.normal(50.0, 20.0, n), "x1": rng.normal(size=n)})
    frame["y"] = frame["x_off"] + 2.0 * frame["x1"] + rng.normal(size=n)
    with_offset = ta.StaticTAM("y ~ offset(x_off) + l(x1)").fit(frame)
    shifted = frame.assign(y=frame["y"] - frame["x_off"])
    plain = ta.StaticTAM("y ~ l(x1)").fit(shifted)
    np.random.seed(0)
    from tam.common.utils import _ensure_dummies
    frame, shifted = (_ensure_dummies(d.copy(), with_offset.group_col_, with_offset.date_col_) for d in (frame, shifted))
    a = compute_bootstrap_significance(with_offset, frame, n_boot=40, block_size=10)
    np.random.seed(0)
    b = compute_bootstrap_significance(plain, shifted, n_boot=40, block_size=10)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_an_offset_in_the_location_formula_of_a_distributional_model():
    rng = np.random.default_rng(12)
    n = 1500
    frame = pd.DataFrame({"e": rng.normal(5.0, 1.0, n), "x": rng.uniform(0, 1, n)})
    frame["y"] = frame["e"] + np.sin(4 * frame["x"]) + 0.3 * rng.normal(size=n)
    model = ta.StaticTAM({"mu": "y ~ offset(e) + s(x, k=10)", "sigma": "y ~ l(x)"}, loss={"mu": "l2", "sigma": "gamma"},
                         dist_kwargs={"log_target": False, "tail_family": "normal"}).fit(frame)
    base = pd.DataFrame({"e": [5.0, 7.0], "x": [0.5, 0.5], "y": 0.0})
    median = model.predict_median(base)
    assert median[1] - median[0] == pytest.approx(2.0, abs=1e-6)
