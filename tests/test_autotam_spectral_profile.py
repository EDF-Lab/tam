# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Spectral profile of a target: cycles of the day, week and year, their orders, interactions, and no false cycle in noise.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.spectral_profile import profile_series, spectral_profile, steps_per_day


def _hourly(n_days=120, seed=0, weekday_profile=False, noise=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(24 * n_days)
    hour = t % 24
    day = t // 24
    y = 10 + 3 * np.sin(2 * np.pi * hour / 24) + 1.5 * np.sin(4 * np.pi * hour / 24 + 1) + 0.8 * np.sin(6 * np.pi * hour / 24 + 2)
    y = y + 2 * np.sin(2 * np.pi * t / 168)
    if weekday_profile:
        y = y + (day % 7 >= 5) * 3 * np.sin(2 * np.pi * hour / 24)
    return pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(t), freq="h"), "y": y + rng.normal(scale=noise, size=len(t))})


def test_a_daily_cycle_of_three_harmonics_and_a_weekly_cycle_are_found_and_no_year():
    profile = spectral_profile(_hourly(), "y", "ds")
    assert profile["cycles"]["day"]["order"] == 3
    assert "week" in profile["cycles"] and "year" not in profile["cycles"]


def test_a_weekday_dependent_daily_profile_sets_the_interaction_flag():
    assert spectral_profile(_hourly(weekday_profile=True), "y", "ds")["interactions"]["hour_x_weekday"] is True
    assert spectral_profile(_hourly(), "y", "ds")["interactions"]["hour_x_weekday"] is False


def test_a_daily_series_with_zero_sundays_has_a_weekly_cycle():
    rng = np.random.default_rng(2)
    n = 460
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    y = 50 + 8 * (dates.dayofweek.to_numpy() == 5) + rng.normal(scale=2, size=n)
    y[dates.dayofweek == 6] = 0.0
    assert "week" in spectral_profile(pd.DataFrame({"ds": dates, "y": y}), "y", "ds")["cycles"]


def test_a_trending_monthly_series_without_seasonality_has_no_cycle_and_a_small_autocorrelation():
    rng = np.random.default_rng(5)
    n = 120
    y = np.cumsum(rng.normal(size=n)) * 0.2 + np.arange(n) * 0.5
    profile = spectral_profile(pd.DataFrame({"ds": pd.date_range("2010-01-01", periods=n, freq="MS"), "y": y}), "y", "ds")
    assert not profile["cycles"]
    assert abs(profile_series(y, 1 / 30.4)["acf"]["year"] or 0.0) < 0.3


def test_white_noise_gives_no_peak_in_nearly_every_series():
    false_positive = 0
    for seed in range(100):
        values = np.random.default_rng(seed).normal(size=1500)
        orders = profile_series(values, 24.0)["orders"]
        false_positive += bool(orders)
    assert false_positive <= 2


def test_the_profile_is_the_same_whatever_follows_the_training_rows():
    frame = _hourly()
    cut = frame.iloc[: 24 * 100]
    assert spectral_profile(cut, "y", "ds") == spectral_profile(pd.concat([cut]), "y", "ds")
    assert profile_series(cut["y"].to_numpy(), 24.0) == profile_series(np.r_[cut["y"].to_numpy()], 24.0)


def test_groups_are_sampled_evenly_and_a_cycle_needs_half_of_them():
    pieces = []
    for g in range(6):
        piece = _hourly(60, seed=g)
        if g >= 4:
            piece["y"] = np.random.default_rng(g).normal(size=len(piece))
        pieces.append(piece.assign(g=g))
    profile = spectral_profile(pd.concat(pieces), "y", "ds", group_col="g", max_groups=6)
    assert profile["cycles"]["day"]["share"] > 0.5


def test_steps_per_day_from_the_median_step():
    assert steps_per_day(pd.Series(pd.date_range("2024-01-01", periods=10, freq="h"))) == 24.0
    assert steps_per_day(pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))) == 1.0
