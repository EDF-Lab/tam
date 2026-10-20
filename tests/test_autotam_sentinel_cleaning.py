# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Missing-data codes and impossible readings (opt-in ``clean_sentinels``): found on the training rows, replaced by the seasonal median before the
lags are built, and applied unchanged to the rows to forecast.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.sentinel_cleaning import SentinelCleaner, find_codes, physical_range

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def test_round_negative_codes_that_repeat_are_found_and_real_negatives_are_not():
    values = np.concatenate([np.random.default_rng(0).normal(10, 3, 500), [-9999.0, -9999.0, -200.0, -200.0, -15.0, -15.0, -3.2]])
    assert find_codes(values) == [-9999.0, -200.0]
    assert find_codes(np.array([1.0, 2.0, 3.0, 4.0, 5.0, -9999.0])) == []            # one occurrence is not a code
    assert find_codes(np.random.default_rng(1).normal(-20, 10, 300)) == []              # a cold winter is not a code


def test_the_physical_range_is_the_one_given_or_a_celsius_temperature_or_a_direction():
    celsius = np.random.default_rng(2).normal(12, 8, 100)
    assert physical_range("Etmp", celsius, None) == (-60.0, 60.0)
    assert physical_range("Etmp", np.random.default_rng(2).normal(285, 8, 100), None) is None          # Kelvin: left alone
    assert physical_range("wind_dir", np.random.default_rng(3).uniform(0, 360, 100), None) == (0.0, 360.0)
    assert physical_range("load", celsius, None) is None
    assert physical_range("load", celsius, (0, 5)) == (0.0, 5.0)


def _hourly(days=60, seed=1):
    rng = np.random.default_rng(seed)
    n = 24 * days
    hour = np.arange(n) % 24
    temp = 10 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal(scale=0.5, size=n)
    y = 100 + 20 * np.sin(2 * np.pi * hour / 24) + 2 * temp + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "temp": temp, "y": y})
    return frame


def test_the_masked_cells_take_the_median_of_their_hour_and_only_those_cells_change():
    frame = _hourly()
    dirty = frame.copy()
    dirty.loc[[30, 55, 200], "y"] = -9999.0
    dirty.loc[[31, 80], "temp"] = -273.0
    dirty.loc[100, "temp"] = -15.0                                                      # a real cold reading stays
    cleaner = SentinelCleaner(["y", "temp"], "ds", None).fit(dirty)
    clean = cleaner.apply(dirty)
    assert clean["y"].min() > 0 and clean["temp"].min() > -60
    assert clean.loc[100, "temp"] == -15.0
    untouched = ~dirty.index.isin([30, 55, 200])
    np.testing.assert_array_equal(clean.loc[untouched, "y"].to_numpy(), dirty.loc[untouched, "y"].to_numpy())
    hour_of_30 = frame["ds"].dt.hour[30]
    expected = dirty.loc[(dirty["ds"].dt.hour == hour_of_30) & (dirty["y"] != -9999.0), "y"].median()
    assert clean.loc[30, "y"] == pytest.approx(expected)
    assert cleaner.report(dirty)["y"]["masked"] == 3 and cleaner.report(dirty)["temp"]["masked"] == 2


def test_a_missing_target_to_forecast_stays_missing_and_the_input_is_not_changed():
    frame = _hourly()
    dirty = frame.copy()
    dirty.loc[[10, 11, 12], "y"] = -9999.0
    cleaner = SentinelCleaner(["y"], "ds", None).fit(dirty)
    future = frame.iloc[-6:].copy()
    future["y"] = np.nan
    out = cleaner.apply(future)
    assert out["y"].isna().all()
    before = dirty.copy()
    cleaner.apply(dirty)
    pd.testing.assert_frame_equal(dirty, before)


def test_the_rare_negatives_of_a_target_that_is_never_negative_are_codes_only_for_the_target():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=2000, freq="D"), "y": rng.uniform(5, 50, 2000), "z": rng.uniform(5, 50, 2000)})
    frame.loc[[3, 700, 1500], ["y", "z"]] = -120.0
    cleaner = SentinelCleaner(["y", "z"], "ds", None, negative_columns=["y"]).fit(frame)
    assert cleaner.rules["y"]["negatives"] and "z" not in cleaner.rules
    half = frame.copy()
    half["y"] = rng.normal(0, 5, 2000)
    assert "y" not in SentinelCleaner(["y"], "ds", None, negative_columns=["y"]).fit(half).rules


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(temp)", n_experts=2, pop_size=6, use_opera=False, seed=2, lags=[24], export_dir=str(directory), **options)


def test_the_lags_carry_no_code_when_the_cleaning_is_on_and_do_when_it_is_off(tmp_path_factory):
    frame = _hourly(days=70)
    dirty = frame.copy()
    dirty.loc[np.arange(100, 1500, 97), "y"] = -9999.0
    clean_model = _model(tmp_path_factory.mktemp("clean"), clean_sentinels=True).fit(dirty.iloc[:24 * 60], date_col="ds", expansions=EXPANSIONS)
    plain_model = _model(tmp_path_factory.mktemp("plain")).fit(dirty.iloc[:24 * 60], date_col="ds", expansions=EXPANSIONS)
    lag = "y_lag_24"
    assert min(f[lag].min() for f in (clean_model.ctx.df_fit, clean_model.ctx.df_dev, clean_model.ctx.df_val)) > -200
    assert min(f[lag].min() for f in (plain_model.ctx.df_fit, plain_model.ctx.df_dev, plain_model.ctx.df_val)) <= -9999
    assert clean_model.sentinel_report_["y"]["masked"] > 0 and plain_model.sentinel_report_ == {}


def test_the_forecast_is_better_with_the_cleaning_when_the_history_has_codes(tmp_path_factory):
    frame = _hourly(days=70)
    dirty = frame.copy()
    dirty.loc[np.arange(100, 1440, 61), "y"] = -9999.0
    train, test = dirty.iloc[:24 * 60], frame.iloc[24 * 60:24 * 66]
    errors = {}
    for label, flag in (("clean", True), ("plain", False)):
        model = _model(tmp_path_factory.mktemp(label), clean_sentinels=flag).fit(train, date_col="ds", expansions=EXPANSIONS)
        out = model.predict(test, date_col="ds")
        errors[label] = float(np.sqrt(np.mean((out[model.trained_experts[0]["name"]].to_numpy() - test["y"].to_numpy()) ** 2)))
    assert errors["clean"] < 0.5 * errors["plain"], errors


def test_the_default_leaves_the_data_alone(tmp_path):
    frame = _hourly(days=40)
    model = _model(tmp_path).fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert model.ctx.sentinel_cleaner is None and model.sentinel_report_ == {}
