# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``AdaptiveTAM`` on the base of a non-Gaussian loss: the windows are fitted on the linear predictor, ``eta = eta_base + Phi theta``, with the
link and the variance of the base (the base predictions enter as an offset), so the adapted forecast keeps the support of the family
(counts and positive means stay non-negative) and a multiplicative drift is tracked on the log scale.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta

N, BASE_ROWS, WINDOW, UPDATE = 1800, 400, 120, 15


def _poisson_drift(seed=0, n=N, zero_streak=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = rng.normal(0.0, 1.0, n)
    drift = np.where(t < BASE_ROWS, 0.0, 0.9 * np.sin(2 * np.pi * (t - BASE_ROWS) / 700.0) + 0.6)  # level moves after the base
    mean = np.exp(1.0 + 0.5 * x + drift)
    y = rng.poisson(mean).astype(float)
    if zero_streak is not None:
        y[zero_streak[0]:zero_streak[1]] = 0.0
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x": x, "y": y})


def _base(df, loss="poisson"):
    return ta.StaticTAM("y ~ l(x)", date_col="date", loss=loss).fit(df.iloc[:BASE_ROWS])


def _adaptive(df, loss="poisson", **kwargs):
    return ta.AdaptiveTAM(base_model=_base(df, loss), adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                          training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0, **kwargs)


def _deviance(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return float(np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))))


def test_a_gaussian_base_does_not_engage_the_glm_mode():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=600, freq="D"), "x": rng.normal(size=600)})
    df["y"] = 1.0 + 2.0 * df["x"] + rng.normal(size=600)
    gaussian = ta.StaticTAM("y ~ l(x)", date_col="date").fit(df.iloc[:300])
    model = ta.AdaptiveTAM(base_model=gaussian, adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=10,
                           training_window_periods=60, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    assert model.glm_strategy_ is None
    assert model.adaptive_model_._offset_features_ == []


@pytest.mark.parametrize("loss", ["poisson", "negative_binomial", "tweedie"])
def test_forecasts_stay_non_negative_through_a_long_streak_of_zeros(loss):
    df = _poisson_drift(zero_streak=(900, 1100))
    model = _adaptive(df, loss)
    frame = model.predict_online(df)
    adapted = frame["AdaptedEstimatedy"].to_numpy()
    assert np.all(np.isfinite(adapted)) and np.all(adapted >= 0.0)


def test_the_adapted_forecast_tracks_a_multiplicative_drift_better_than_the_base_and_than_a_gaussian_correction():
    df = _poisson_drift(seed=1)
    glm = _adaptive(df)
    frame = glm.predict_online(df)
    late = frame.index[BASE_ROWS + 2 * WINDOW:]
    y = frame.loc[late, "y"].to_numpy()
    glm_deviance = _deviance(y, frame.loc[late, "AdaptedEstimatedy"].to_numpy())
    base_deviance = _deviance(y, frame.loc[late, "Estimatedy"].to_numpy())
    # the same correction fitted by least squares on the residuals of a Gaussian base of the same formula
    gaussian_base = ta.StaticTAM("y ~ l(x)", date_col="date").fit(df.iloc[:BASE_ROWS])
    gaussian = ta.AdaptiveTAM(base_model=gaussian_base, adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                              training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    gaussian_frame = gaussian.predict_online(df)
    gaussian_deviance = _deviance(y, np.clip(gaussian_frame.loc[late, "AdaptedEstimatedy"].to_numpy(), 1e-9, None))
    assert glm_deviance < 0.6 * base_deviance
    assert glm_deviance < gaussian_deviance


def test_without_drift_the_correction_is_small():
    df = _poisson_drift(seed=2)
    df.loc[BASE_ROWS:, "y"] = np.random.default_rng(5).poisson(np.exp(1.0 + 0.5 * df.loc[BASE_ROWS:, "x"].to_numpy()))
    frame = _adaptive(df).predict_online(df)
    late = frame.index[BASE_ROWS + WINDOW:]
    ratio = frame.loc[late, "AdaptedEstimatedy"].to_numpy() / frame.loc[late, "Estimatedy"].to_numpy()
    assert abs(np.median(ratio) - 1.0) < 0.1


def test_rows_before_the_first_window_keep_the_base_forecast():
    df = _poisson_drift()
    frame = _adaptive(df).predict_online(df)
    first = frame.index[:50]
    assert np.allclose(frame.loc[first, "AdaptedEstimatedy"], frame.loc[first, "Estimatedy"])


def test_fit_then_predict_uses_the_frozen_state_on_the_response_scale():
    df = _poisson_drift(seed=3)
    model = _adaptive(df).fit(df.iloc[:1500])
    out = model.predict(df.iloc[1500:])
    assert np.all(out["AdaptedEstimatedy"].to_numpy() > 0.0)
    assert out["AdaptedEstimatedy"].corr(out["y"]) > 0.3


def test_quantiles_are_discrete_ordered_and_non_negative():
    df = _poisson_drift(seed=4)
    quantiles = _adaptive(df).predict_quantiles(df, (0.1, 0.5, 0.9))
    q = quantiles.to_numpy()
    assert np.all(q >= 0.0) and np.allclose(q, np.round(q))
    assert np.all(q[:, 0] <= q[:, 1]) and np.all(q[:, 1] <= q[:, 2])
    late = slice(BASE_ROWS + 2 * WINDOW, None)
    y = df["y"].to_numpy()[late]
    inside = (y >= q[late, 0]) & (y <= q[late, 2])
    assert 0.7 < float(inside.mean()) <= 1.0


def test_clipping_to_the_training_range_is_refused_with_a_glm_base():
    df = _poisson_drift()
    with pytest.raises(ValueError, match="clip_to_train_range"):
        _adaptive(df, clip_to_train_range=True)


def test_gamma_and_binomial_bases_keep_their_support():
    rng = np.random.default_rng(7)
    t = np.arange(N)
    x = rng.normal(size=N)
    shift = np.where(t < BASE_ROWS, 0.0, 0.8)
    positive = rng.gamma(4.0, np.exp(0.5 + 0.4 * x + shift) / 4.0)
    probability = 1.0 / (1.0 + np.exp(-(0.3 * x + shift)))
    dates = pd.date_range("2020-01-01", periods=N, freq="D")
    for loss, y in (("gamma", positive), ("binomial", rng.binomial(1, probability).astype(float))):
        df = pd.DataFrame({"date": dates, "x": x, "y": y})
        frame = _adaptive(df, loss).predict_online(df)
        adapted = frame["AdaptedEstimatedy"].to_numpy()
        assert np.all(np.isfinite(adapted)) and np.all(adapted > 0.0)
        if loss == "binomial":
            assert np.all(adapted < 1.0)


def test_the_formula_may_name_the_base_target_itself():
    df = _poisson_drift(seed=6)
    base = _base(df)
    residual = ta.AdaptiveTAM(base_model=base, adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                              training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    named = ta.AdaptiveTAM(base_model=base, adaptive_formula="y ~ l(effect_x)", update_interval_periods=UPDATE,
                           training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    a, b = residual.predict_online(df), named.predict_online(df)
    assert np.allclose(a["AdaptedEstimatedy"], b["AdaptedEstimatedy"], equal_nan=True)
    assert np.allclose(a["Estimatedy"], b["Estimatedy"], equal_nan=True)  # the base forecast is not overwritten


def test_grid_search_returns_a_model_that_keeps_the_glm_mode():
    df = _poisson_drift(seed=7)
    model = ta.AdaptiveTAM(base_model=_base(df), adaptive_formula="Residualy ~ l(effect_x, ap='pen')", update_interval_periods=UPDATE,
                           training_window_periods=WINDOW, steps_per_period=1, date_col="date")
    best = model.grid_search_fit(df, {"pen": [-6.0, -2.0]})
    assert best.glm_strategy_ is not None
    assert np.all(best.predictions_["AdaptedEstimatedy"].to_numpy() >= 0.0)
