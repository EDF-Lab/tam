# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Predictions of groups that do not share their timestamps are aligned to the rows asked for by date and group, not by index.
"""

import numpy as np
import pandas as pd

import tam as ta
from tam.model.autotam.alignment import aligned_values


def _weekday_frame(n=280, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n)
    frame = pd.DataFrame({"ds": dates, "g": dates.dayofweek.to_numpy(), "x": rng.normal(size=n)})
    frame["y"] = 3 + frame["x"] * (1 + frame["g"]) + rng.normal(scale=0.1, size=n)
    return frame


def test_the_engine_returns_another_index_for_groups_on_disjoint_dates_and_the_alignment_restores_the_rows():
    frame = _weekday_frame()
    train, test = frame.iloc[:200], frame.iloc[200:].copy()
    model = ta.StaticTAM("y ~ l(x)", group_col="g", date_col="ds").fit(train)
    out = model.predict(test)
    assert not out.index.equals(test.index)                       # the reason for the helper
    values = aligned_values(out, test, "Estimatedy", "ds", "g")
    assert len(values) == len(test) and np.isfinite(values).all()
    # the value of each row is the prediction of its own date and group
    check = test[["ds", "g"]].merge(out[["ds", "g", "Estimatedy"]], on=["ds", "g"], how="left")
    np.testing.assert_array_equal(values, check["Estimatedy"].to_numpy())


def test_the_values_are_read_in_place_when_the_indexes_agree():
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=4), "g": [0, 0, 1, 1]}, index=[10, 11, 12, 13])
    predictions = frame.assign(v=[1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(aligned_values(predictions, frame, "v", "ds", "g"), [1.0, 2.0, 3.0, 4.0])


def test_a_shuffled_result_is_matched_on_the_keys():
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=4), "g": [0, 1, 0, 1]})
    predictions = frame.assign(v=[10.0, 20.0, 30.0, 40.0]).iloc[[3, 1, 0, 2]].reset_index(drop=True)
    np.testing.assert_array_equal(aligned_values(predictions, frame, "v", "ds", "g"), [10.0, 20.0, 30.0, 40.0])
