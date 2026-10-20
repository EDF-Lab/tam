# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Differenced targets: the increments are the target of the fit, the levels are kept apart, and the forecast adds them back step by step.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import differencing as dif


def _series(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.5, 1.5, size=n))


def _frame(values, start="2023-01-01", group=None):
    out = pd.DataFrame({"ds": pd.date_range(start, periods=len(values), freq="D"), "y": values})
    if group is not None:
        out["g"] = group
    return out


def test_the_target_of_the_training_frames_is_the_increment_and_the_first_rows_are_dropped():
    y = _series()
    frames = {"fit": _frame(y[:40]), "dev": _frame(y[40:50], "2023-02-10"), "val": _frame(y[50:], "2023-02-20")}
    out, levels = dif.difference_training(frames, "y", "ds", None, 3)
    np.testing.assert_allclose(out["fit"]["y"].to_numpy(), y[3:40] - y[:37])
    np.testing.assert_allclose(out["dev"]["y"].to_numpy(), y[40:50] - y[37:47])          # the lag crosses the frames
    np.testing.assert_allclose(out["val"]["y"].to_numpy(), y[50:] - y[47:57])
    assert len(out["fit"]) == 37 and len(levels) == 60 and dif.LEVEL in levels.columns
    assert dif.LEVEL not in out["fit"].columns          # the level is never a column the model could read


def test_the_lag_is_taken_inside_each_group():
    a, b = _series(30, 1), _series(30, 2) * 10
    frame = pd.concat([_frame(a, group="a"), _frame(b, group="b")], ignore_index=True)
    out, _ = dif.difference_training({"fit": frame}, "y", "ds", "g", 2)
    got = out["fit"]
    np.testing.assert_allclose(got.loc[got["g"] == "a", "y"].to_numpy(), a[2:] - a[:-2])
    np.testing.assert_allclose(got.loc[got["g"] == "b", "y"].to_numpy(), b[2:] - b[:-2])


def test_adding_back_the_true_increments_of_forecast_rows_gives_the_true_levels_inside_the_horizon_and_beyond():
    y = _series(70)
    train, future = _frame(y[:60]), _frame(y[60:], "2023-03-02")
    _, levels = dif.difference_training({"fit": train}, "y", "ds", None, 3)
    truth = pd.Series(y[60:] - y[57:67])                          # the true increments y_t - y_{t-3}, for t = 60..69
    forecast_rows = future.assign(y=np.nan)
    got = dif.add_back(pd.DataFrame({"model": truth.to_numpy()}), forecast_rows, levels, "y", "ds", None, 3)
    np.testing.assert_allclose(got["model"].to_numpy(), y[60:])    # rows 63.. are rebuilt from forecast levels, not from observations


def test_an_observed_level_in_the_test_frame_is_the_base_of_the_rows_that_follow():
    y = _series(70)
    train = _frame(y[:60])
    _, levels = dif.difference_training({"fit": train}, "y", "ds", None, 3)
    future = _frame(y[60:], "2023-03-02")
    increments = pd.DataFrame({"m": np.zeros(10)})               # a model that predicts no increment at all
    seen = dif.add_back(increments, future, levels, "y", "ds", None, 3)
    hidden = dif.add_back(increments, future.assign(y=np.nan), levels, "y", "ds", None, 3)
    np.testing.assert_allclose(seen["m"].to_numpy()[:3], y[57:60])          # the level three steps before, observed
    np.testing.assert_allclose(seen["m"].to_numpy()[3:], y[60:67])          # then the observed levels of the frame itself
    np.testing.assert_allclose(hidden["m"].to_numpy(), np.tile(y[57:60], 4)[:10])   # without them, the last three levels repeat


def test_the_test_frame_target_becomes_the_increment_where_it_is_known():
    y = _series(70)
    train = _frame(y[:60])
    _, levels = dif.difference_training({"fit": train}, "y", "ds", None, 3)
    future = _frame(y[60:], "2023-03-02")
    future.loc[5:, "y"] = np.nan
    out = dif.difference_test(future, levels, "y", "ds", None, 3)
    np.testing.assert_allclose(out["y"].to_numpy()[:5], y[60:65] - y[57:62])
    assert out["y"].iloc[5:].isna().all()


def test_a_test_row_on_a_training_date_takes_the_training_level():
    y = _series(40)
    train = _frame(y)
    _, levels = dif.difference_training({"fit": train}, "y", "ds", None, 2)
    inside = _frame(y[30:], "2023-01-31").assign(y=np.nan)
    out = dif.difference_test(inside, levels, "y", "ds", None, 2)
    np.testing.assert_allclose(out["y"].to_numpy(), y[30:] - y[28:38])


def test_a_bad_lag_or_a_missing_date_column_is_refused():
    frame = {"fit": _frame(_series(20))}
    with pytest.raises(ValueError, match="positive whole number"):
        dif.difference_training(frame, "y", "ds", None, 0)
    with pytest.raises(ValueError, match="date column"):
        dif.difference_training(frame, "y", None, None, 1)


# --------------------------------------------------------------------------- #
# A search on increments
# --------------------------------------------------------------------------- #

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _cumulative(n=260, seed=4):
    rng = np.random.default_rng(seed)
    step = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n), "y": 100 + np.cumsum(step)})


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    frame = _cumulative()
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=2, pop_size=4, use_opera=True, seed=3, difference=1,
                    export_dir=str(tmp_path_factory.mktemp("difference")))
    model.fit(frame.iloc[:230], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_the_forecast_of_a_cumulative_series_is_a_level_that_keeps_rising(fitted):
    model, frame = fitted
    test = frame.iloc[230:237].assign(y=np.nan)
    out = model.predict(test, date_col="ds")[model.final_model_column_]
    assert out.notna().all() and (np.diff(out.to_numpy()) > 0).all()
    last = frame["y"].iloc[229]
    truth = frame["y"].iloc[230:237].to_numpy()
    assert np.abs(out.to_numpy() - truth).mean() < 0.5 * np.abs(last - truth).mean()      # far better than repeating the last level


def test_the_difference_of_the_search_is_recorded_and_the_online_path_is_refused(fitted):
    model, frame = fitted
    assert model.difference_ == 1
    with pytest.raises(NotImplementedError, match="increments"):
        model.predict_online(frame.iloc[230:235], date_col="ds")


def test_auto_differences_a_cumulative_series_and_leaves_a_stationary_one_alone(tmp_path):
    cumulative = AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=1, pop_size=4, use_opera=False, seed=3, difference="auto", export_dir=str(tmp_path))
    cumulative.fit(_cumulative().iloc[:120], date_col="ds", expansions=EXPANSIONS)
    assert cumulative.difference_ == 7          # the horizon: the smallest lag
    rng = np.random.default_rng(1)
    flat = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=120), "x1": rng.normal(size=120)})
    flat["y"] = 10 + flat["x1"] + rng.normal(scale=0.3, size=120)
    stationary = AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=1, pop_size=4, use_opera=False, seed=3, difference="auto", export_dir=str(tmp_path))
    stationary.fit(flat, date_col="ds", expansions=EXPANSIONS)
    assert stationary.difference_ is None


def test_a_bad_difference_is_refused(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1)", difference="weekly", export_dir=str(tmp_path))
    with pytest.raises(ValueError, match="difference must be"):
        model.fit(_cumulative().iloc[:100], date_col="ds", expansions=EXPANSIONS)


def test_auto_also_differences_a_random_walk_that_is_not_monotone(tmp_path):
    rng = np.random.default_rng(3)
    n = 220
    walk = np.cumsum(rng.normal(size=n))                       # goes up and down: not cumulative, but a unit root
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": rng.normal(size=n), "y": 50 + walk})
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=1, pop_size=4, use_opera=False, seed=3, difference="auto", export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert model.difference_ == 7


def test_the_unit_root_vote_needs_most_of_the_groups():
    rng = np.random.default_rng(5)
    walks = [np.cumsum(rng.normal(size=120)) for _ in range(3)]
    flats = [rng.normal(size=120) for _ in range(3)]
    def panel(series):
        return pd.concat([pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=120), "g": i, "y": v}) for i, v in enumerate(series)])
    assert AutoTAM._has_unit_root(panel(walks), "y", "ds", "g") is True
    assert AutoTAM._has_unit_root(panel(flats), "y", "ds", "g") is False
    assert AutoTAM._has_unit_root(panel(walks[:1] + flats[:2]), "y", "ds", "g") is False
