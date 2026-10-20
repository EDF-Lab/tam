# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The inactivity gate (opt-in): series that never opened, positions of the season that are always zero and rows declared closed are forecast zero,
point and quantiles; the zeros before a group opened are left out of the fit.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.inactivity import InactivityGate, drop_leading_zeros

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _panel(n=120):
    rng = np.random.default_rng(1)
    rows = []
    for shop in (0, 1, 2):
        x = rng.normal(size=n)
        y = 50 + 5 * x + rng.normal(scale=1.0, size=n)
        if shop == 1:
            y = np.zeros(n)                                  # never opened
        if shop == 2:
            y[:80] = 0.0                                     # opened late
        rows.append(pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "shop": shop, "x1": x, "y": y, "Open": 1.0}))
    return pd.concat(rows, ignore_index=True)


def test_the_leading_zeros_of_each_group_are_dropped_and_an_all_zero_group_is_kept():
    out = drop_leading_zeros(_panel(), "y", "shop", "ds")
    assert len(out[out["shop"] == 2]) == 40 and len(out[out["shop"] == 1]) == 120 and len(out[out["shop"] == 0]) == 120
    series = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=6), "y": [0, 0, 3, 0, 4, 5.0]})
    assert drop_leading_zeros(series, "y", None, "ds")["y"].tolist() == [3, 0, 4, 5]
    assert drop_leading_zeros(series.assign(y=1.0), "y", None, "ds") is not None


def test_the_gate_finds_the_unopened_group_the_cutoff_and_the_closed_rows():
    panel = _panel()
    panel.loc[(panel["shop"] == 0) & (panel["ds"] > "2023-04-18"), "y"] = 0.0           # stops at the end of the history
    gate = InactivityGate("y", "ds", "shop", {"Open": 0}).fit(panel)
    assert gate.unopened == {1} and 0 in gate.inactive_at_cutoff
    test = pd.DataFrame({"ds": pd.date_range("2023-06-01", periods=3).tolist() * 3, "shop": [0] * 3 + [1] * 3 + [2] * 3, "Open": [1, 0, 1] * 3})
    assert gate.mask(test).tolist() == [False, True, False, True, True, True, False, True, False]


def test_the_night_of_a_solar_series_is_a_zero_position():
    n = 24 * 60
    hour = np.arange(n) % 24
    y = np.where((hour >= 7) & (hour <= 18), 100.0 + np.random.default_rng(2).normal(size=n), 0.0)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "y": y})
    gate = InactivityGate("y", "ds", None).fit(frame)
    day = pd.DataFrame({"ds": pd.date_range("2023-03-05", periods=24, freq="h")})
    assert gate.mask(day).tolist() == [h < 7 or h > 18 for h in range(24)]


def test_apply_sets_every_forecast_column_of_the_gated_rows_to_zero():
    gate = InactivityGate("y", "ds", None, {"Open": 0})
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=3), "Open": [1, 0, 1]})
    values = pd.DataFrame({"a": [5.0, 6.0, 7.0], "q0.1": [4.0, 5.0, 6.0]})
    out = gate.apply(values, frame)
    assert out["a"].tolist() == [5.0, 0.0, 7.0] and out["q0.1"].tolist() == [4.0, 0.0, 6.0] and values["a"].tolist() == [5.0, 6.0, 7.0]


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=True, seed=3, export_dir=str(directory), **options)


def test_a_fit_forecasts_zero_for_an_unopened_group_and_a_closed_row_point_and_quantiles(tmp_path):
    panel = _panel()
    train, test = panel[panel["ds"] < "2023-04-15"], panel[panel["ds"] >= "2023-04-15"].copy()
    test.loc[test.index[5], "Open"] = 0.0
    model = _model(tmp_path, inactivity_gate=True, closed_when={"Open": 0}).fit(train, date_col="ds", group_col="shop", expansions=EXPANSIONS)
    assert model.gate_.unopened == {1}
    out = model.predict(test, date_col="ds")
    column = model.final_model_column_
    assert (out.loc[(test["shop"] == 1).to_numpy(), column] == 0).all() and out.loc[5, column] == 0
    open_rows = ((test["shop"] == 0) & (test["Open"] == 1)).to_numpy()
    assert (out.loc[open_rows, column] > 10).all()
    quantiles = model.predict_quantiles(test, date_col="ds")
    assert (quantiles.loc[(test["shop"] == 1).to_numpy()] == 0).all().all()


def test_without_the_option_nothing_is_gated_and_the_zeros_stay_in_the_fit(tmp_path):
    panel = _panel()
    train = panel[panel["ds"] < "2023-04-15"]
    model = _model(tmp_path).fit(train, date_col="ds", group_col="shop", expansions=EXPANSIONS)
    assert model.gate_ is None and model.inactive_groups_ == []
    assert len(model.ctx.df_fit) + len(model.ctx.df_dev) + len(model.ctx.df_val) >= len(train) - 5
