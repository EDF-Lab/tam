# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The data profiler keeps the rows of groups that are regular on different phases of the same grid (a group per day of the week of a daily
series), and gives the same frame as the common-grid resample whenever the groups share it.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.data_profiler import DataProfiler


def _profile(frame, group_col="g"):
    return DataProfiler()._enforce_time_continuity(frame.copy(), "ds", group_col, is_fit=True)


def _weekday_groups(n=140):
    """One row a day, the group being the day of the week: seven groups, each stepping by seven days on its own phase."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ds": dates, "g": dates.dayofweek, "x": rng.normal(size=n), "y": rng.normal(size=n)})


def test_groups_on_other_phases_keep_all_their_rows_and_values():
    frame = _weekday_groups()
    out = _profile(frame)
    assert len(out) == len(frame) and out[["x", "y"]].notna().all().all()
    merged = out.merge(frame, on=["ds", "g"], suffixes=("", "_in"))
    assert len(merged) == len(frame)
    np.testing.assert_array_equal(merged["x"], merged["x_in"])
    np.testing.assert_array_equal(merged["y"], merged["y_in"])


def test_groups_on_a_shared_grid_give_the_same_frame_as_the_common_grid_resample(monkeypatch):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    frame = pd.concat([pd.DataFrame({"ds": dates, "g": g, "x": rng.normal(size=60), "y": rng.normal(size=60)}) for g in ("a", "b", "c")])
    frame.loc[frame.index[5], "x"] = np.nan                                   # a missing value is filled the same way on both paths
    regular = _profile(frame)
    monkeypatch.setattr(DataProfiler, "_groups_are_regular", staticmethod(lambda data, group_col: False))
    resampled = _profile(frame)
    pd.testing.assert_frame_equal(regular.reset_index(drop=True), resampled.reset_index(drop=True))


def test_a_group_with_a_gap_is_still_filled_on_the_common_grid():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    frame = pd.concat([pd.DataFrame({"ds": dates, "g": g, "x": rng.normal(size=40)}) for g in ("a", "b")])
    frame = frame.drop(frame.index[(frame["g"] == "a") & (frame["ds"] == dates[10])])
    out = _profile(frame)
    assert len(out) == 80 and out["x"].notna().all()
