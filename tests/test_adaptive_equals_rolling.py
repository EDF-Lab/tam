# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
AdaptiveTAM is the naive operational loop: for every window, ``StaticTAM.fit`` on the training rows, then ``predict`` on the next
rows. The two must give the same forecasts (here to 1e-8), whatever the formula, regularisation, horizon and number of groups.

``ta.rolling_windows`` cuts the windows. They are anchored on the start of the data; the rows before the first forecast are NaN.
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta

N, L, W = 120, 40, 10

FORMULAS = {
    "l": "y ~ l(x)",
    "s": "y ~ s(z, k=6)",
    "f": "y ~ f(t)",
    "c": "y ~ c(k, n_cat=3, topo='nominal')",
    "l+s+te": "y ~ l(x) + s(z, k=5) + te(s(x, k=4), s(z, k=4))",
}


def _data(n_groups=1, n=N, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        t = np.arange(n)
        x = 1.0 + t / n * 4.0 + rng.normal(0, 0.3, n)                   # the range of x drifts upwards
        z = np.sin(t / 15.0) * (1 + t / n) + rng.normal(0, 0.1, n)
        k = t % 3
        slope = 1.0 + 1.5 * (t > n // 2)                                  # a regime change
        y = slope * x + np.sin(3 * z) + 0.5 * k + g + rng.normal(0, 0.2, n)
        frames.append(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "g": g, "x": x, "z": z,
                                    "t": (t % 12).astype(float), "k": k, "y": y}))
    return pd.concat(frames, ignore_index=True)


def _adaptive(formula, ap, horizon, n_groups, **kwargs):
    return ta.AdaptiveTAM(adaptive_formula=formula, update_interval_periods=W, training_window_periods=L, steps_per_period=1,
                          horizon_steps=horizon, group_col="g" if n_groups > 1 else None, date_col="date", default_alpha_p=ap, **kwargs)


def _loop(df, formula, ap, horizon, n_groups):
    """The operational reference: StaticTAM refit on every window."""
    group_col = "g" if n_groups > 1 else None
    out = pd.Series(np.nan, index=df.index)
    for train_rows, forecast_rows in ta.rolling_windows(df, L, W, 1, horizon, group_col=group_col, date_col="date"):
        model = ta.StaticTAM(formula=formula, group_col=group_col, date_col="date", default_alpha_p=ap).fit(df.loc[train_rows])
        forecast = model.predict(df.loc[forecast_rows])
        out.loc[forecast.index] = forecast["Estimatedy"].to_numpy()
    return out


@pytest.mark.parametrize("n_groups", [1, 3])
@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("ap", [-6.0, 0.0])
@pytest.mark.parametrize("name", list(FORMULAS))
def test_adaptive_equals_the_rolling_static_loop(name, ap, horizon, n_groups):
    df = _data(n_groups)
    adaptive = _adaptive(FORMULAS[name], ap, horizon, n_groups).predict_online(df)
    reference = _loop(df, FORMULAS[name], ap, horizon, n_groups)
    np.testing.assert_array_equal(adaptive["Estimatedy"].isna().to_numpy(), reference.isna().to_numpy())
    both = reference.notna().to_numpy()
    assert both.sum() > 0
    np.testing.assert_allclose(adaptive["Estimatedy"].to_numpy()[both], reference.to_numpy()[both], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("horizon", [1, 3])
def test_rows_before_the_first_forecast_are_nan_and_the_rest_are_forecast(horizon):
    df = _data(3)
    out = _adaptive(FORMULAS["l"], -6.0, horizon, 3).predict_online(df)
    first = L + horizon - 1
    for _, frame in out.groupby("g"):
        frame = frame.sort_values("date")
        assert frame["Estimatedy"].iloc[:first].isna().all() and frame["AdaptedEstimatedy"].iloc[:first].isna().all()
        assert frame["Estimatedy"].iloc[first:].notna().all()


def test_with_a_base_model_the_rows_before_the_first_window_keep_the_base_forecast():
    df = _data(1)
    base = ta.StaticTAM(formula="y ~ l(x)", date_col="date").fit(df.iloc[:60])
    model = ta.AdaptiveTAM(base_model=base, adaptive_formula="Residualy ~ l(z)", update_interval_periods=W, training_window_periods=L,
                           steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    out = model.predict_online(df)
    warm = out.iloc[:L]
    assert warm["EstimatedResidualy"].isna().all() and not out["EstimatedResidualy"].iloc[L:].isna().any()
    np.testing.assert_allclose(warm["AdaptedEstimatedy"].to_numpy(), warm["Estimatedy"].to_numpy(), rtol=0, atol=0)
    assert out["AdaptedEstimatedy"].notna().all()


def test_windows_start_on_the_data_and_the_last_one_is_cut():
    df = _data(1, n=105)
    windows = list(ta.rolling_windows(df, L, W, 1, 1, date_col="date"))
    assert [(len(a), len(b)) for a, b in windows] == [(L, W)] * 6 + [(L, 5)]
    assert windows[0][0][0] == 0 and windows[0][1][0] == L and windows[-1][1][-1] == 104


def test_no_look_ahead_a_forecast_ignores_the_rows_after_it():
    df = _data(3)
    model = _adaptive(FORMULAS["l+s+te"], -6.0, 1, 3)
    before = model.predict_online(df)["Estimatedy"].to_numpy()
    changed = df.copy()
    late = changed["date"] > pd.Timestamp("2020-01-01") + pd.Timedelta(days=70)
    rng = np.random.default_rng(1)
    changed.loc[late, ["x", "z", "y"]] = rng.normal(size=(int(late.sum()), 3)) * 50
    after = _adaptive(FORMULAS["l+s+te"], -6.0, 1, 3).predict_online(changed)["Estimatedy"].to_numpy()
    early = (~late).to_numpy() & ~np.isnan(before)
    assert early.sum() > 0
    np.testing.assert_allclose(after[early], before[early], rtol=1e-8, atol=1e-8)


def test_every_window_trains_on_the_full_normalised_range():
    """Per-window normalisation: the training rows of every window map exactly to [-1, 1]."""
    df = _data(2)
    model = _adaptive(FORMULAS["s"], -6.0, 1, 2)
    model.prepare_simulation(df)
    x_train = model.simulation_data_[0]                                        # (groups, windows, L, features)
    np.testing.assert_allclose(x_train.amin(dim=2).numpy(), -1.0, atol=1e-12)
    np.testing.assert_allclose(x_train.amax(dim=2).numpy(), 1.0, atol=1e-12)


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("n_groups", [1, 3])
def test_fit_then_predict_is_the_last_window_static_model(horizon, n_groups):
    df = _data(n_groups)
    history = df[df["date"] < df["date"].min() + pd.Timedelta(days=95)]
    future = df.drop(history.index)
    formula = FORMULAS["l+s+te"]
    model = _adaptive(formula, -6.0, horizon, n_groups).fit(history)
    predicted = model.predict(future)["Estimatedy"].to_numpy()

    # the window an operational refit holds after the last row of the history: starts at the last multiple of W
    n = history[history["g"] == 0].shape[0]
    start = L + horizon - 1 + ((n - (L + horizon - 1)) // W) * W
    group_col = "g" if n_groups > 1 else None
    rows = history.sort_values(["g", "date"]).groupby("g").apply(
        lambda f: f.iloc[start - (horizon - 1) - L: start - (horizon - 1)], include_groups=False).reset_index(level=0).sort_index()
    reference = ta.StaticTAM(formula=formula, group_col=group_col, date_col="date", default_alpha_p=-6.0).fit(rows)
    np.testing.assert_allclose(predicted, reference.predict(future)["Estimatedy"].to_numpy(), rtol=1e-8, atol=1e-8)


def test_predict_does_not_renormalise_on_the_prediction_data():
    """The old predict() normalised on the rows to predict: the same rows shifted gave another curve."""
    df = _data(1)
    history, future = df.iloc[:100], df.iloc[100:]
    model = _adaptive(FORMULAS["s"], -6.0, 1, 1).fit(history)
    inside = model.predict(future)["Estimatedy"].to_numpy()
    wide = model.predict(pd.concat([future, future.assign(z=future["z"] + 2.0, date=future["date"] + pd.Timedelta(days=100))],
                                   ignore_index=True))["Estimatedy"].to_numpy()[: len(future)]
    np.testing.assert_allclose(wide, inside, rtol=1e-9, atol=1e-9)


def test_no_hidden_clip_by_default_and_clip_to_the_window_range_on_request():
    df = _data(1)
    df.loc[df["date"] > pd.Timestamp("2020-01-01") + pd.Timedelta(days=80), "x"] += 4.0      # x leaves the trained range
    free = _adaptive(FORMULAS["l"], -6.0, 1, 1).predict_online(df)["Estimatedy"]
    clipped = _adaptive(FORMULAS["l"], -6.0, 1, 1, clip_to_train_range=True).predict_online(df)["Estimatedy"]
    assert free.max() > df["y"].max()                  # extrapolated freely: no whole-period clip
    assert (free != clipped).any() and clipped.max() <= df["y"].max()


def test_filled_rows_are_never_trained_or_forecast():
    df = _data(3)
    short = df.drop(df[(df["g"] == 2) & (df["date"] > pd.Timestamp("2020-01-01") + pd.Timedelta(days=89))].index)
    out = _adaptive(FORMULAS["l"], -6.0, 1, 3).predict_online(short)
    assert len(out) == len(short)
    reference = _loop(short, FORMULAS["l"], -6.0, 1, 3)
    both = reference.notna().to_numpy()
    np.testing.assert_allclose(out["Estimatedy"].to_numpy()[both], reference.to_numpy()[both], rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(out["Estimatedy"].isna().to_numpy(), reference.isna().to_numpy())
