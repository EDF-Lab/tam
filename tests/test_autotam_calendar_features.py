# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Calendar features and country holidays: positions on a cycle derived from the date column, seeds from the spectral profile, no linear
calendar term, and nothing changed without the arguments.
"""

import re
import sys

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import calendar_features as cf
from tam.model.autotam.auto_tam import AutoTAM


def _daily(n=900, seed=0, holiday_effect=False):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2022-01-01", periods=n, freq="D")
    toy = (ds.dayofyear.to_numpy() - 1) / 365.0
    y = 100 + 8 * np.sin(2 * np.pi * toy) + 6 * (ds.dayofweek.to_numpy() >= 5) + rng.normal(scale=1.0, size=n)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"ds": ds, "x1": x1, "y": y + 0.5 * x1})


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #

def test_the_calendar_names_follow_the_sampling_step():
    assert cf.calendar_names(24.0) == ["tod", "toy", "dow"]
    assert cf.calendar_names(1.0) == ["toy", "dow"]
    assert cf.calendar_names(1 / 7) == ["toy"]


def test_the_positions_are_fractions_of_the_day_and_of_the_year_and_the_weekday():
    dates = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-07-01 12:00", "2024-12-31 18:00"]))
    out = cf.calendar_columns(dates, ["tod", "toy", "dow"])
    assert out["tod"].tolist() == [0.0, 0.5, 0.75]
    assert out["toy"].iloc[0] == 0.0 and 0.0 < out["toy"].iloc[1] < 1.0 and out["toy"].iloc[2] < 1.0
    assert out["dow"].tolist() == [0.0, 0.0, 1.0]


def test_a_column_the_user_already_has_is_kept():
    frame = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "toy": [9.0, 9.0, 9.0]})
    added = cf.add_calendar(frame, "ds", {"names": ["toy", "dow"]})
    assert added == ["dow"] and frame["toy"].tolist() == [9.0] * 3


# --------------------------------------------------------------------------- #
# Holidays
# --------------------------------------------------------------------------- #

def test_france_2024_has_eleven_holidays_and_a_bridge_day_on_the_tenth_of_may():
    pytest.importorskip("holidays")
    dates = pd.Series(pd.date_range("2024-01-01", "2024-12-31"))
    out = cf.holiday_columns(dates, "FR")
    assert int(out["holiday"].sum()) == 11
    assert (out.loc[out["holiday"] == 1, "days_to_holiday"] == 0).all()
    assert out.loc[dates == "2024-05-10", "bridge_day"].iloc[0] == 1.0
    assert out["days_to_holiday"].max() <= cf.HOLIDAY_DISTANCE_CAP and out["days_since_holiday"].max() <= cf.HOLIDAY_DISTANCE_CAP


def test_a_missing_holidays_package_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "holidays", None)
    with pytest.raises(ImportError, match="tam-ml\[calendar\]"):
        cf.holiday_columns(pd.Series(pd.date_range("2024-01-01", periods=3)), "FR")


def test_an_unknown_country_is_refused():
    pytest.importorskip("holidays")
    with pytest.raises(ValueError, match="no calendar"):
        cf.holiday_columns(pd.Series(pd.date_range("2024-01-01", periods=3)), "ZZ")


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #

def test_seed_formulas_follow_the_cycles_present():
    profile = {"cycles": {"day": {"order": 3}, "week": {"order": 5}, "year": {"order": 4}}, "interactions": {"hour_x_weekday": True}}
    seeds = cf.calendar_seed_formulas("y", profile, ["tod", "toy", "dow"])
    assert seeds[0] == "y ~ f(tod, m=3, cyclic=True, period=(0, 1)) + f(dow, m=3, cyclic=True, period='auto') + f(toy, m=4, cyclic=True, period=(0, 1))"
    assert len(seeds) == 2 and "te(" in seeds[1]
    assert cf.calendar_seed_formulas("y", {"cycles": {}, "interactions": {}}, ["tod"]) == []
    assert cf.calendar_seed_formulas("y", {"cycles": {"day": {"order": 2}}, "interactions": {}}, ["toy"]) == []


# --------------------------------------------------------------------------- #
# A search
# --------------------------------------------------------------------------- #

def _model(tmp_path, **kwargs):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=False, seed=3, export_dir=str(tmp_path), **kwargs)


@pytest.fixture(scope="module")
def calendar_fit(tmp_path_factory):
    return _model(tmp_path_factory.mktemp("calendar"), calendar_features=True).fit(
        _daily().iloc[:800], date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})


def test_daily_data_get_toy_and_dow_but_no_tod(calendar_fit):
    columns = set(calendar_fit.ctx.df_all_aug.columns)
    assert {"toy", "dow"} <= columns and "tod" not in columns


def test_the_profile_finds_the_weekly_and_yearly_cycles_and_they_are_seeded(calendar_fit):
    assert {"week", "year"} <= set(calendar_fit.calendar_profile_["cycles"])
    assert any("Seed" in str(row.get("Model_Name")) for row in calendar_fit.chronological_test_log)   # tried, whether or not it survives the selection


def test_no_formula_of_the_search_has_a_linear_calendar_term(calendar_fit):
    linear = re.compile(r"(?<![A-Za-z])(l|s|p|t|w|n|rbf|lt)\(\s*(toy|tod|dow)")
    formulas = [f for fs in calendar_fit.architectures_.values() for f in fs]
    formulas += [str(e.get(key) or "") for e in calendar_fit.trained_experts for key in ("formula", "dynamic_formula", "base_formula")]
    assert [f for f in formulas if linear.search(f)] == []


def test_predict_builds_the_calendar_columns_of_the_forecast_rows(calendar_fit):
    out = calendar_fit.predict(_daily().iloc[800:807], date_col="ds")
    column = calendar_fit.trained_experts[0]["name"]
    assert len(out) == 7 and np.isfinite(out[column]).all()


def test_without_the_arguments_nothing_is_added_and_the_results_are_identical(tmp_path):
    frame = _daily().iloc[:300]
    expansions = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}
    one = _model(tmp_path).fit(frame, date_col="ds", expansions=expansions)
    two = _model(tmp_path, calendar_features=False, country=None).fit(frame, date_col="ds", expansions=expansions)
    assert not one.ctx.calendar and one.architectures_ == two.architectures_
    assert not {"toy", "dow", "tod"} & set(one.ctx.df_all_aug.columns)


def test_calendar_features_need_a_date_column(tmp_path):
    with pytest.raises(ValueError, match="date column"):
        _model(tmp_path, calendar_features=True).fit(_daily().iloc[:200].drop(columns="ds"))


def test_a_declared_country_adds_the_holiday_columns(tmp_path):
    pytest.importorskip("holidays")
    model = _model(tmp_path, country="FR").fit(_daily().iloc[:300], date_col="ds",
                                              expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert set(cf.HOLIDAY_COLUMNS) <= set(model.ctx.df_all_aug.columns) and "toy" not in model.ctx.df_all_aug.columns


# --------------------------------------------------------------------------- #
# Columns the frame already has
# --------------------------------------------------------------------------- #

def test_a_position_the_frame_already_has_under_another_name_is_not_added_again():
    dates = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    frame = pd.DataFrame({"ds": dates, "hour": dates.hour.to_numpy(), "day_of_week": dates.dayofweek.to_numpy()})
    spec = {"names": ["tod", "toy", "dow"]}
    added = cf.add_calendar(frame, "ds", spec, skip_equivalent=True)
    assert added == ["toy"] and spec["keep"] == ["toy"]
    assert "tod" not in frame.columns and "dow" not in frame.columns


def test_the_forecast_rows_follow_the_columns_kept_at_the_fit():
    dates = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    fit_frame = pd.DataFrame({"ds": dates, "hour": dates.hour.to_numpy()})
    spec = {"names": ["tod", "toy", "dow"]}
    cf.add_calendar(fit_frame, "ds", spec, skip_equivalent=True)
    ahead = pd.DataFrame({"ds": pd.date_range("2024-03-01", periods=3, freq="h")})
    assert cf.add_calendar(ahead, "ds", spec) == ["toy", "dow"] and list(ahead.columns) == ["ds", "toy", "dow"]


def test_a_frame_without_equivalent_columns_gets_all_of_them():
    dates = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    frame = pd.DataFrame({"ds": dates, "x1": np.random.default_rng(0).normal(size=len(dates))})
    spec = {"names": ["tod", "toy", "dow"]}
    assert cf.add_calendar(frame, "ds", spec, skip_equivalent=True) == ["tod", "toy", "dow"]
