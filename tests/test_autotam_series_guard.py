# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The guard per series (opt-in): a group whose validation RMSE is smaller for the pooled seasonal floor or the seasonal anchor than for the final model
is forecast by that expert.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.series_guard import SINGLE_SERIES, apply_guard, guarded_series

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def test_only_the_groups_where_a_guard_is_better_are_guarded():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    final = np.array([1.0, 2.0, 9.0, 9.0, 5.0, 6.0])                           # exact on a and c, wrong on b
    guards = {"floor": np.array([1.5, 2.5, 3.1, 4.1, 5.5, 6.5]), "anchor": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    assert guarded_series(truth, final, guards, ["a", "a", "b", "b", "c", "c"]) == {"b": "floor"}


def test_a_tie_does_not_replace_the_final_model_and_missing_values_do_not_count():
    truth = np.array([1.0, 2.0, np.nan, 4.0])
    assert guarded_series(truth, truth.copy(), {"floor": truth.copy()}, ["a", "a", "b", "b"]) == {}
    final = np.array([1.0, 2.0, 3.0, np.nan])
    assert guarded_series(truth, final, {"floor": np.array([1.0, 2.0, 3.0, 4.0])}, ["a", "a", "b", "b"]) == {}


def test_a_single_series_is_one_group():
    truth = np.arange(5.0)
    out = guarded_series(truth, truth + 3, {"floor": truth + 1}, None)
    assert out == {SINGLE_SERIES: "floor"}


def test_the_guard_replaces_the_final_column_on_the_rows_of_the_guarded_groups_only():
    frame = pd.DataFrame({"final": [1.0, 1.0, 1.0, 1.0], "floor": [7.0, 7.0, np.nan, 7.0]})
    out = apply_guard(frame, "final", {"b": "floor"}, ["a", "a", "b", "b"])
    assert out["final"].tolist() == [1.0, 1.0, 1.0, 7.0]                       # no floor value on the third row: kept
    assert frame["final"].tolist() == [1.0] * 4                                 # the input is not changed
    assert apply_guard(frame, "final", {}, ["a"] * 4) is frame
    assert apply_guard(frame, "missing", {"b": "floor"}, ["a"] * 4) is frame


def _panel(n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for site, noise in ((0, 0.2), (1, 3.0), (2, 0.5)):
        x1 = rng.normal(size=n)
        y = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + (2 * x1 if site == 0 else 0.0) + rng.normal(scale=noise, size=n)
        rows.append(pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x1, "y": y, "site": site}))
    return pd.concat(rows, ignore_index=True)


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=True, seed=5, lags=[7], export_dir=str(directory), **options)


def test_the_guard_needs_a_guard_expert():
    with pytest.raises(ValueError, match="guard expert"):
        AutoTAM("y ~ AutoPipe(x1)", series_guard=True, seasonal_anchor=False)


def test_a_panel_is_fitted_with_the_guard_and_the_groups_are_listed(tmp_path):
    frame = _panel()
    model = _model(tmp_path, pooled_floor=True, series_guard=True).fit(frame[frame["ds"] < "2023-05-01"], date_col="ds", group_col="site", expansions=EXPANSIONS)
    assert isinstance(model.guarded_series_, dict) and {int(k) for k in model.guarded_series_} <= {0, 1, 2}
    guard_experts = {e["name"] for e in model.trained_experts if e.get("island") == "Floor"}
    assert set(model.guarded_series_.values()) <= guard_experts


def test_a_guarded_group_is_forecast_by_its_guard_expert_and_the_others_by_the_final_model(tmp_path):
    frame = _panel()
    train, test = frame[frame["ds"] < "2023-05-01"], frame[frame["ds"] >= "2023-05-01"].iloc[:90]
    model = _model(tmp_path, pooled_floor=True, series_guard=True).fit(train, date_col="ds", group_col="site", expansions=EXPANSIONS)
    floor = next(e["name"] for e in model.trained_experts if e.get("island") == "Floor")
    model.guarded_series_ = {1: floor}                                        # forced: what matters here is what predict() does with it
    guarded = model.predict(test, date_col="ds")
    model.guarded_series_ = {}
    plain = model.predict(test, date_col="ds")
    rows_b = (test["site"] == 1).to_numpy()
    column = model.final_model_column_
    np.testing.assert_allclose(guarded.loc[rows_b, column].to_numpy(), plain.loc[rows_b, floor].to_numpy())
    np.testing.assert_allclose(guarded.loc[~rows_b, column].to_numpy(), plain.loc[~rows_b, column].to_numpy())


def test_without_the_option_nothing_is_guarded(tmp_path):
    frame = _panel()
    model = _model(tmp_path, pooled_floor=True).fit(frame[frame["ds"] < "2023-05-01"], date_col="ds", group_col="site", expansions=EXPANSIONS)
    assert model.series_guard is False and model.guarded_series_ == {}
