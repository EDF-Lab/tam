# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Unit tests for ``tam.model.autotam.data_profiler.DataProfiler``.

Covers stateful train-fold profiling (NA dropping, IQR bound learning,
group-aware and global clipping), the leak-free ``transform`` for unseen data,
cold-start group fallback, and time-continuity enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.data_profiler import DataProfiler


def _cross_sectional_frame():
    rng = np.random.default_rng(0)
    temp = rng.normal(15, 3, 50)
    temp[0] = 500.0  # a severe outlier to be clipped
    return pd.DataFrame({
        "load": rng.normal(100, 10, 50),   # target
        "temp": temp,                       # feature with an outlier
        "junk": [np.nan] * 35 + list(rng.normal(0, 1, 15)),  # 70% NA -> dropped
    })


def test_profile_drops_high_na_columns_and_learns_bounds():
    profiler = DataProfiler(na_threshold=0.40)
    cleaned, metadata = profiler.profile_and_clean(
        _cross_sectional_frame(), config={"targets": ["load"]}
    )
    assert "junk" not in cleaned.columns
    assert metadata["dropped_na_cols"] == ["junk"]
    assert "temp" in metadata["bounds"]["global"]


def test_profile_clips_outliers():
    profiler = DataProfiler(na_threshold=0.40)
    cleaned, _ = profiler.profile_and_clean(
        _cross_sectional_frame(), config={"targets": ["load"]}
    )
    # The injected 500.0 outlier is clipped down to the upper IQR bound.
    assert cleaned["temp"].max() < 500.0


def test_profile_empty_frame_raises():
    profiler = DataProfiler()
    with pytest.raises(ValueError, match="empty or None"):
        profiler.profile_and_clean(pd.DataFrame(), config={"targets": ["load"]})


def test_transform_applies_learned_bounds():
    profiler = DataProfiler(na_threshold=0.40)
    profiler.profile_and_clean(_cross_sectional_frame(), config={"targets": ["load"]})

    new = pd.DataFrame({"load": [100.0, 110.0], "temp": [9999.0, 14.0]})
    transformed = profiler.transform(new, config={"targets": ["load"]})
    upper_bound = profiler.metadata["bounds"]["global"]["temp"][1]
    assert transformed["temp"].max() <= upper_bound


def test_transform_empty_frame_returns_empty():
    profiler = DataProfiler()
    profiler.profile_and_clean(_cross_sectional_frame(), config={"targets": ["load"]})
    assert profiler.transform(pd.DataFrame(), config={"targets": ["load"]}).empty


def test_group_aware_bounds_are_learned():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "load": rng.normal(100, 10, 60),
        "temp": rng.normal(15, 3, 60),
        "site": ["A"] * 30 + ["B"] * 30,
    })
    profiler = DataProfiler()
    _, metadata = profiler.profile_and_clean(df, config={"targets": ["load"]}, group_col="site")
    assert "A" in metadata["bounds"] and "B" in metadata["bounds"]


def test_cold_start_group_falls_back_to_global_bounds():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({
        "load": rng.normal(100, 10, 40),
        "temp": rng.normal(15, 3, 40),
        "site": ["A"] * 20 + ["B"] * 20,
    })
    profiler = DataProfiler()
    profiler.profile_and_clean(train, config={"targets": ["load"]}, group_col="site")

    # 'Z' was never seen in training -> clipping uses the global bounds.
    unseen = pd.DataFrame({"load": [100.0], "temp": [9999.0], "site": ["Z"]})
    transformed = profiler.transform(unseen, config={"targets": ["load"]})
    assert transformed["temp"].iloc[0] <= profiler.metadata["bounds"]["global"]["temp"][1]


def test_time_series_continuity_sets_frequency():
    dates = pd.date_range("2022-01-01", periods=12, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "load": np.arange(12, dtype=float),
        "temp": np.linspace(0, 5, 12),
    })
    profiler = DataProfiler()
    cleaned, metadata = profiler.profile_and_clean(
        df, config={"targets": ["load"]}, date_col="ds"
    )
    assert metadata["is_time_series"] is True
    assert "delta_t" in metadata
    assert not cleaned.empty


def _grouped_frame_with_a_gap():
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    rows = []
    for site, offset in (("a", 0.0), ("b", 100.0)):
        for i, d in enumerate(dates):
            if site == "b" and i == 3:
                continue  # a missing day in one group
            rows.append({"ds": d, "site": site, "load": offset + i, "temp": float(i)})
    return pd.DataFrame(rows)


def test_grouped_continuity_raises_no_future_warning():
    """``groupby().resample()`` used to operate on the grouping column, which a future pandas excludes."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        DataProfiler().profile_and_clean(_grouped_frame_with_a_gap(), config={"targets": ["load"]}, date_col="ds", group_col="site")


def test_grouped_continuity_fills_the_gap_of_each_group_and_keeps_the_columns():
    cleaned, _ = DataProfiler().profile_and_clean(
        _grouped_frame_with_a_gap(), config={"targets": ["load"]}, date_col="ds", group_col="site"
    )
    assert list(cleaned.columns) == ["ds", "load", "temp", "site"]  # the grouping column is restored last
    assert len(cleaned) == 12  # 6 days for each of the two groups
    b = cleaned[cleaned["site"] == "b"].reset_index(drop=True)
    assert list(b["ds"]) == list(pd.date_range("2022-01-01", periods=6, freq="D"))
    assert b.loc[3, "load"] == 102.0  # the missing day is the previous value of its own group, not of the other
    assert (cleaned["site"].value_counts() == 6).all()

