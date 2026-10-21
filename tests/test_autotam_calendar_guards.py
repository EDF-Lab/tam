# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Calendar guards (opt-in ``calendar_guards``): the time of year only with two years of dates and a step finer than a year, the holidays only at a
step of a day or finer, and the day before, the day after and the year-end week as holiday columns.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.calendar_features import extra_holiday_flags, guarded_calendar_names

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _dates(periods, freq):
    return pd.Series(pd.date_range("2020-01-01", periods=periods, freq=freq))


def test_the_time_of_year_needs_two_years_of_dates():
    kept, dropped = guarded_calendar_names(["tod", "toy", "dow"], _dates(24 * 31, "h"), 24.0)
    assert kept == ["tod", "dow"] and "toy" in dropped and "year" in dropped["toy"]
    kept, dropped = guarded_calendar_names(["toy", "dow"], _dates(3 * 365, "D"), 1.0)
    assert kept == ["toy", "dow"] and dropped == {}


def test_the_time_of_year_is_not_offered_at_a_yearly_step():
    kept, dropped = guarded_calendar_names(["toy"], _dates(40, "YS"), 1 / 365.25)
    assert kept == [] and "constant" in dropped["toy"]


def test_a_quarterly_series_with_enough_years_keeps_its_time_of_year():
    assert guarded_calendar_names(["toy"], _dates(40, "QS"), 1 / 91.3)[0] == ["toy"]


def test_the_extra_flags_mark_the_day_before_the_day_after_and_the_year_end_week():
    day = pd.Series(pd.date_range("2023-12-20", "2024-01-05"))
    holidays = pd.DatetimeIndex(["2023-12-25", "2024-01-01"])
    flags = extra_holiday_flags(day, holidays).set_index(day)
    assert flags.loc["2023-12-24", "day_before_holiday"] == 1 and flags.loc["2023-12-26", "day_after_holiday"] == 1
    assert flags.loc["2023-12-31", "day_before_holiday"] == 1 and flags.loc["2024-01-02", "day_after_holiday"] == 1
    assert flags.loc["2023-12-22", "day_before_holiday"] == 0 and flags.loc["2023-12-27", "day_after_holiday"] == 0
    assert flags["year_end_week"].loc["2023-12-23"] == 0 and flags["year_end_week"].loc["2023-12-24":"2024-01-02"].eq(1).all()
    assert flags["year_end_week"].loc["2024-01-03"] == 0


def _hourly(days):
    rng = np.random.default_rng(1)
    n = 24 * days
    hour = np.arange(n) % 24
    y = 50 + 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "x1": rng.normal(size=n), "y": y})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=6, use_opera=False, seed=3, export_dir=str(directory), calendar_features=True, **options)


def test_a_short_hourly_history_has_no_time_of_year_with_the_guard_and_has_one_without(tmp_path_factory):
    frame = _hourly(40)
    guarded = _model(tmp_path_factory.mktemp("g"), calendar_guards=True).fit(frame, date_col="ds", expansions=EXPANSIONS)
    plain = _model(tmp_path_factory.mktemp("p")).fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert "toy" not in guarded.ctx.calendar["columns"] and "tod" in guarded.ctx.calendar["columns"]
    assert "toy" in plain.ctx.calendar["columns"]


@pytest.mark.slow
def test_the_guard_leaves_a_long_daily_history_as_it_was(tmp_path_factory):
    rng = np.random.default_rng(2)
    n = 800
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    frame["y"] = 10 + np.sin(2 * np.pi * np.arange(n) / 365.25) + 0.1 * frame["x1"] + rng.normal(scale=0.1, size=n)
    guarded = _model(tmp_path_factory.mktemp("g"), calendar_guards=True).fit(frame, date_col="ds", expansions=EXPANSIONS)
    plain = _model(tmp_path_factory.mktemp("p")).fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert guarded.ctx.calendar["columns"] == plain.ctx.calendar["columns"]


def test_the_holidays_of_a_country_are_dropped_at_a_monthly_step_and_the_extra_flags_come_with_them(tmp_path):
    pytest.importorskip("holidays")
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame({"ds": pd.date_range("2014-01-01", periods=n, freq="MS"), "x1": rng.normal(size=n)})
    frame["y"] = 10 + frame["x1"] + rng.normal(scale=0.1, size=n)
    model = _model(tmp_path, calendar_guards=True, country="FR").fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert not any(c in model.ctx.calendar["columns"] for c in ("holiday", "day_before_holiday"))
