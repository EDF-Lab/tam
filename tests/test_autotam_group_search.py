# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Group search: a series given without a group column is also fitted with a group derived from its dates (one group per period of the day on
sub-daily data, one per day of the week on daily data), and the version with the smaller validation error is kept.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.group_search import GROUP_COLUMN, MAX_GROUPS, add_group, derived_group

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _dates(periods, freq):
    return pd.Series(pd.date_range("2023-01-01", periods=periods, freq=freq))


@pytest.mark.parametrize("periods, freq, expected", [
    (24 * 90, "h", ("time_of_day", 24)),
    (96 * 90, "15min", ("time_of_day", 96)),
    (48 * 90, "30min", ("time_of_day", 48)),
])
def test_the_grouping_the_dates_allow(periods, freq, expected):
    assert derived_group(_dates(periods, freq)) == expected


@pytest.mark.parametrize("periods, freq", [(200, "D"), (200, "W"), (60, "MS"), (1440 * 5, "min"), (24 * 10, "h"), (10, "D")])
def test_no_grouping_for_a_coarse_fine_or_short_series(periods, freq):
    assert derived_group(_dates(periods, freq)) is None


def test_no_grouping_for_repeated_or_irregular_dates():
    dates = _dates(24 * 90, "h")
    assert derived_group(pd.concat([dates, dates])) is None
    rng = np.random.default_rng(0)
    irregular = pd.Series(pd.Timestamp("2023-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 90 * 86400, 2000)), unit="s")).drop_duplicates()
    assert derived_group(irregular) is None
    assert MAX_GROUPS == 96


def test_the_group_column_is_the_period_of_the_day_or_the_day_of_week():
    hourly = pd.DataFrame({"ds": pd.date_range("2023-01-01 00:00", periods=48, freq="h")})
    assert add_group(hourly, "ds", "time_of_day", 24)[GROUP_COLUMN].tolist() == list(range(24)) * 2
    daily = pd.DataFrame({"ds": pd.date_range("2023-01-02", periods=14, freq="D")})   # a Monday
    assert add_group(daily, "ds", "day_of_week", 7)[GROUP_COLUMN].tolist() == list(range(7)) * 2
    assert GROUP_COLUMN not in hourly.columns   # a copy


def _hourly(days=110, seed=3):
    """One response per hour of the day: the slope of x1 changes with the hour, which one ungrouped model with x1 alone cannot follow."""
    rng = np.random.default_rng(seed)
    n = 24 * days
    hour = np.arange(n) % 24
    x1 = rng.normal(size=n)
    slope = 1.0 + 4.0 * np.sin(2 * np.pi * hour / 24) ** 2 * np.where(hour < 12, 1.0, -1.0)
    y = 50 + slope * x1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "x1": x1, "y": y})


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    frame = _hourly()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=5, group_search=True, lags=[24],
                    export_dir=str(tmp_path_factory.mktemp("group_search")))
    model.fit(frame.iloc[:24 * 90], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_both_versions_are_scored_and_the_better_one_is_kept(fitted):
    model, _ = fitted
    assert set(model.group_scores_) == {"ungrouped", "time_of_day"}
    kept = model.derived_group_ is not None
    assert kept == (model.group_scores_["time_of_day"] < model.group_scores_["ungrouped"])
    if kept:
        assert model.derived_group_ == {"column": GROUP_COLUMN, "kind": "time_of_day", "count": 24}
        assert model.ctx.group_col == GROUP_COLUMN


def test_a_frame_without_the_group_column_is_forecast_and_keeps_its_rows(fitted):
    model, frame = fitted
    test = frame.iloc[24 * 90:24 * 95]
    out = model.predict(test, date_col="ds")
    assert len(out) == len(test) and GROUP_COLUMN not in test.columns
    assert np.isfinite(out[model.final_model_column_].to_numpy()).all()


def test_a_given_group_column_is_not_searched(tmp_path):
    frame = _hourly(days=70)
    frame["site"] = "a"
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search=True, lags=[24], export_dir=str(tmp_path))
    model.fit(frame.iloc[:24 * 60], date_col="ds", group_col="site", expansions=EXPANSIONS)
    assert model.derived_group_ is None and model.group_scores_ == {}


def test_a_series_that_allows_no_grouping_is_fitted_as_before(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n, freq="MS"), "x1": rng.normal(size=n)})
    frame["y"] = 5 + frame["x1"] + rng.normal(scale=0.2, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search=True, export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert model.derived_group_ is None and model.group_scores_ == {} and model.trained_experts


def test_the_lags_and_the_horizon_of_the_group_version_are_counted_in_cycles(tmp_path):
    frame = _hourly(days=110)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search=True, lags=[24, 48, 30], horizon=24,
                    export_dir=str(tmp_path))
    model.fit(frame.iloc[:24 * 90], date_col="ds", expansions=EXPANSIONS)
    assert set(model.group_scores_) == {"ungrouped", "time_of_day"}
    if model.derived_group_ is not None:                                  # the grouped version won: its lags are 1 and 2 cycles
        assert sorted(model.data_manager.user_lags) == [1, 2] and model.data_manager.horizon == 1
    else:
        assert sorted(model.data_manager.user_lags) == [24, 30, 48]


def test_a_series_whose_lags_are_no_whole_number_of_cycles_is_not_searched(tmp_path):
    frame = _hourly(days=70)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search=True, lags=[5],
                    export_dir=str(tmp_path))
    model.fit(frame.iloc[:24 * 60], date_col="ds", expansions=EXPANSIONS)
    assert model.derived_group_ is None and model.group_scores_ == {}


def test_always_fits_only_the_grouped_version(tmp_path):
    frame = _hourly(days=110)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search="always", lags=[24],
                    export_dir=str(tmp_path))
    model.fit(frame.iloc[:24 * 90], date_col="ds", expansions=EXPANSIONS)
    assert model.derived_group_ == {"column": GROUP_COLUMN, "kind": "time_of_day", "count": 24}
    assert model.group_scores_ == {} and model.ctx.group_col == GROUP_COLUMN and model.data_manager.user_lags == [1]
    assert len(model.predict(frame.iloc[24 * 90:24 * 92], date_col="ds")) == 48


def test_always_falls_back_to_the_plain_fit_when_the_dates_allow_no_group(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n, freq="MS"), "x1": rng.normal(size=n)})
    frame["y"] = 5 + frame["x1"] + rng.normal(scale=0.2, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search="always", export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert model.derived_group_ is None and model.trained_experts


def test_an_unknown_group_search_value_is_refused(tmp_path):
    with pytest.raises(ValueError, match="group_search"):
        AutoTAM("y ~ AutoPipe(x1)", group_search="sometimes", export_dir=str(tmp_path))
