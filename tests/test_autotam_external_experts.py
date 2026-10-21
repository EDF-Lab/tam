# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
External experts in the Apex aggregation (forecasts made by the caller, given by date) and the pruning of the members that keep no weight
(``apex_prune``).
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.external_experts import look_up, normalise_external

EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}


def _frame(n=220):
    rng = np.random.default_rng(11)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * x1 + x2 ** 2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x1, "x2": x2, "y": y})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1, x2)", **{"n_experts": 5, "pop_size": 8, "seed": 4, "dynamic_top": 1, "export_dir": str(directory), **options})


def _truth(frame, noise=0.0):
    values = frame["y"].to_numpy() + np.random.default_rng(3).normal(scale=noise, size=len(frame))
    return pd.Series(values, index=pd.DatetimeIndex(frame["ds"]))


def test_a_series_a_two_level_series_and_a_frame_are_read_by_date():
    dates = pd.date_range("2024-01-01", periods=3)
    series = pd.Series([1.0, 2.0, 3.0], index=dates)
    grouped = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_tuples([(dates[0], "a"), (dates[0], "b")]))
    frame = pd.DataFrame({"ds": dates, "forecast": [4.0, 5.0, 6.0]})
    out = normalise_external({"s": series, "g": grouped, "f": frame}, "ds", "grp")
    assert list(out["s"].columns) == ["ds", "forecast"] and list(out["g"].columns) == ["ds", "grp", "forecast"]
    rows = pd.DataFrame({"ds": [dates[2], dates[0], pd.Timestamp("2030-01-01")]})
    np.testing.assert_array_equal(look_up(rows, out["f"], "ds"), [6.0, 4.0, np.nan])


@pytest.mark.parametrize("name", ["with space", "1abc", "a-b", ""])
def test_a_name_that_cannot_be_written_in_a_formula_is_refused(name):
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        normalise_external({name: pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01"]))}, "ds", None)


def test_the_forecasts_need_the_date_column_and_a_known_form():
    with pytest.raises(ValueError, match="date column"):
        normalise_external({"a": pd.Series([1.0])}, None, None)
    with pytest.raises(ValueError, match="pd.Series"):
        normalise_external({"a": [1.0, 2.0]}, "ds", None)


def test_an_external_expert_equal_to_the_truth_takes_the_weight(tmp_path):
    frame = _frame()
    model = _model(tmp_path).fit(frame, date_col="ds", expansions=EXPANSIONS, external_experts={"oracle": _truth(frame)})
    assert "oracle" in model.weights_top10 and model.weights_top10["oracle"] > 0.5
    assert "oracle" in model.apex_members_
    row = model.experts_.set_index("identifier").loc["oracle"]
    assert row["name"] == "oracle" and row["apex_weight"] > 0.5
    assert model.experts_["apex_weight"].sum() == pytest.approx(1.0)


def test_the_external_forecast_enters_the_final_forecast_at_predict_time(tmp_path):
    frame = _frame()
    model = _model(tmp_path).fit(frame.iloc[:180], date_col="ds", expansions=EXPANSIONS, external_experts={"oracle": _truth(frame.iloc[:180], 1.0)})
    test = frame.iloc[180:]
    column = model.final_model_column_
    with_oracle = model.predict(test, external_experts={"oracle": _truth(test, 1.0)})
    without = model.predict(test)
    error = lambda out: np.abs(out[column].to_numpy() - test["y"].to_numpy()).mean()
    assert np.isfinite(error(with_oracle)) and not np.allclose(with_oracle[column], without[column])      # the external forecast moves the mix
    assert "oracle" not in without.columns and np.isfinite(without[column]).all()   # the others are mixed alone
    assert model.predict(test, external_experts={"oracle": _truth(test, 1.0)}, by_identifier=True)["oracle"].notna().all()


def test_an_external_forecast_missing_on_validation_rows_is_refused(tmp_path):
    frame = _frame()
    with pytest.raises(ValueError, match="missing on"):
        _model(tmp_path).fit(frame, date_col="ds", expansions=EXPANSIONS, external_experts={"half": _truth(frame).iloc[:100]})


def test_an_external_name_that_is_an_expert_name_is_refused(tmp_path):
    frame = _frame()
    with pytest.raises(ValueError, match="already the name of an expert"):
        _model(tmp_path).fit(frame, date_col="ds", expansions=EXPANSIONS, external_experts={"Static_LinearIsland_R1_Prior": _truth(frame)})


def test_external_experts_need_the_apex_aggregation(tmp_path):
    frame = _frame()
    with pytest.raises(ValueError, match="use_opera"):
        _model(tmp_path, use_opera=False).fit(frame, date_col="ds", expansions=EXPANSIONS, external_experts={"a": _truth(frame)})


@pytest.mark.slow
def test_a_zero_threshold_leaves_the_pool_and_the_forecasts_as_they_were(tmp_path_factory):
    frame = _frame()
    plain = _model(tmp_path_factory.mktemp("plain"), pop_size=4).fit(frame, date_col="ds", expansions=EXPANSIONS)
    zero = _model(tmp_path_factory.mktemp("zero"), pop_size=4, apex_prune=0.0).fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert plain.apex_pruned_ == [] and zero.apex_pruned_ == []
    assert plain.weights_top10 == zero.weights_top10
    assert [e["name"] for e in plain.trained_experts] == [e["name"] for e in zero.trained_experts]
    pd.testing.assert_frame_equal(plain.predict(frame.iloc[150:]), zero.predict(frame.iloc[150:]))


def test_pruning_shrinks_the_pool_and_the_members_left_carry_the_weights(tmp_path_factory):
    frame = _frame()
    plain = _model(tmp_path_factory.mktemp("full"), apex_quality_ratio=None).fit(frame, date_col="ds", expansions=EXPANSIONS)
    pruned = _model(tmp_path_factory.mktemp("cut"), apex_quality_ratio=None, apex_prune=0.01).fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert pruned.apex_pruned_ and len(pruned.apex_members_) == len(plain.apex_members_) - len(pruned.apex_pruned_)
    assert set(pruned.apex_pruned_).isdisjoint(pruned.apex_members_)
    assert len(pruned.trained_experts) <= len(plain.trained_experts)
    assert np.isfinite(pruned.predict(frame.iloc[150:])[pruned.final_model_column_]).all()


def test_a_threshold_outside_zero_one_is_refused(tmp_path):
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="apex_prune"):
            _model(tmp_path, apex_prune=bad)
