# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Dynamic experts and ensembles on forecast rows.

The Kalman and adaptive experts forecast rows whose target is not known. The target of those rows stays missing (it is neither a column to
drop the row on nor a value to fill forward), every expert is complete on every row up to the horizon, an ensemble mixes the experts that are
finite on a row, and a frame inside the kept history gets the same columns as one that follows it.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM, finite_weighted_mix, split_forecast_rows

HORIZON = 7


def _series(n=160):
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    df["y"] = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * df["x1"] + rng.normal(scale=0.3, size=n)
    return df


# --------------------------------------------------------------------------- #
# The mix of the experts that are finite on a row
# --------------------------------------------------------------------------- #

def test_a_row_without_missing_experts_is_the_plain_weighted_mean_bit_for_bit():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": [100.0, 200.0, 300.0]})
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    expected = sum(frame[m] * (w / sum(weights.values())) for m, w in weights.items())
    got, dropped = finite_weighted_mix(frame, weights)
    assert np.array_equal(got.to_numpy(), expected.to_numpy())
    assert dropped == 0


def test_a_missing_expert_is_dropped_on_its_rows_and_the_weights_are_renormalised():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, np.nan, 30.0], "c": [100.0, 200.0, 300.0]})
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    got, dropped = finite_weighted_mix(frame, weights)
    assert got.iloc[1] == pytest.approx((2.0 * 0.5 + 200.0 * 0.2) / 0.7)
    assert got.iloc[0] == pytest.approx(1.0 * 0.5 + 10.0 * 0.3 + 100.0 * 0.2)
    assert dropped == 1


def test_a_row_where_every_expert_is_missing_stays_missing():
    frame = pd.DataFrame({"a": [np.nan, 1.0], "b": [np.nan, 2.0]})
    got, dropped = finite_weighted_mix(frame, {"a": 0.5, "b": 0.5})
    assert np.isnan(got.iloc[0]) and got.iloc[1] == pytest.approx(1.5)
    assert dropped == 2


# --------------------------------------------------------------------------- #
# Observed rows and forecast rows
# --------------------------------------------------------------------------- #

def test_the_forecast_rows_are_the_rows_after_the_last_observed_target():
    df = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0, np.nan, np.nan]})
    observed, forecast = split_forecast_rows(df, "y", None)
    assert observed.index.tolist() == [0, 1, 3]          # the gap at 2 is neither observed nor a forecast row
    assert forecast.index.tolist() == [4, 5]


def test_the_forecast_rows_are_found_inside_each_group():
    df = pd.DataFrame({"g": ["a", "b", "a", "b", "a", "b"], "y": [1.0, 1.0, 2.0, np.nan, np.nan, np.nan]})
    observed, forecast = split_forecast_rows(df, "y", "g")
    assert observed.index.tolist() == [0, 1, 2]
    assert forecast.index.tolist() == [3, 4, 5]


def test_a_frame_with_every_target_has_no_forecast_row():
    observed, forecast = split_forecast_rows(pd.DataFrame({"y": [1.0, 2.0, 3.0]}), "y", None)
    assert len(observed) == 3 and forecast.empty


# --------------------------------------------------------------------------- #
# A search with Kalman and adaptive experts
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_dynamic_rows"))
    try:
        df = _series()
        past, future = df.iloc[:150].copy(), df.iloc[150:150 + HORIZON].copy()
        future["y"] = np.nan                       # the target of the forecast rows is not known
        model = AutoTAM("y ~ AutoPipe(x1)", lags=[HORIZON], n_experts=1, pop_size=4, use_opera=True, seed=5)
        model.fit(past, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False})
        yield model, past, future
    finally:
        os.chdir(previous)


def test_the_search_has_kalman_and_adaptive_experts(fitted):
    model, _, _ = fitted
    assert {"kalman", "adaptive"} <= {e["type"] for e in model.trained_experts}


def test_every_expert_and_ensemble_is_complete_on_every_forecast_row(fitted):
    model, _, future = fitted
    out = model.predict(future, date_col="ds")
    incomplete = out.isna().sum()
    assert len(out) == HORIZON
    missing_columns = [e["name"] for e in model.trained_experts if e["name"] not in out.columns]
    assert not missing_columns, f"{len(missing_columns)} experts have no column on the forecast rows (their prediction failed)"
    assert not incomplete[incomplete > 0].to_dict()
    assert model.final_model_column_ in out.columns


def test_the_forecast_rows_alone_equal_the_tail_of_past_plus_future_for_every_expert(fitted):
    model, past, future = fitted
    alone = model.predict(future, date_col="ds")
    full = model.predict(pd.concat([past, future]), date_col="ds").tail(HORIZON)
    shared = [c for c in alone.columns if c in full.columns]
    assert len(shared) == len(alone.columns)
    for column in shared:
        np.testing.assert_allclose(alone[column].to_numpy(), full[column].to_numpy(), rtol=0, atol=1e-9, err_msg=column)


def test_a_frame_inside_the_kept_history_has_the_same_columns_as_the_forecast_rows(fitted):
    model, past, future = fitted
    inside = model.predict(past.tail(20), date_col="ds")
    after = model.predict(future, date_col="ds")
    assert len(inside) == 20
    assert list(inside.columns) == list(after.columns)
    assert inside[model.final_model_column_].notna().all()


def test_the_target_of_the_forecast_rows_is_not_filled_forward(fitted):
    model, past, future = fitted
    frame, _ = model._with_history(future, "ds")
    _, augmented = model.data_manager.transform_test_data(frame, model.ctx)
    assert augmented[model.ctx.target].tail(HORIZON).isna().all()
    assert augmented[model.ctx.target].iloc[:-HORIZON].notna().all()


# --------------------------------------------------------------------------- #
# The horizon argument
# --------------------------------------------------------------------------- #

def test_a_lag_shorter_than_the_horizon_is_refused_at_fit(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[1, 7], horizon=7, n_experts=1, pop_size=4, export_dir=str(tmp_path))
    with pytest.raises(ValueError, match=r"horizon=7.*\[1\]"):
        model.fit(_series(80), date_col="ds")


@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_a_horizon_that_is_not_a_positive_whole_number_is_refused(bad):
    with pytest.raises(ValueError, match="horizon"):
        AutoTAM("y ~ AutoPipe(x1)", horizon=bad)


def test_without_the_horizon_argument_it_is_the_smallest_lag(fitted):
    model, _, _ = fitted
    assert model.ctx.requested_horizon is None
    assert model.ctx.inferred_horizon == min(model.ctx.adaptive_lags) == HORIZON


def test_the_horizon_argument_sets_the_lags_of_the_dynamic_experts(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1)", horizon=3, n_experts=1, pop_size=4, use_opera=False, seed=5, export_dir=str(tmp_path))
    model.fit(_series(120), date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": True, "grid": False})
    assert model.ctx.inferred_horizon == 3
    assert model.ctx.adaptive_lags == [3, 7]


def test_the_week_is_a_lag_of_the_dynamic_experts_only_for_steps_of_a_day_or_less(tmp_path):
    weekly = _series(120).assign(ds=lambda d: pd.date_range("2020-01-05", periods=len(d), freq="W"))
    model = AutoTAM("y ~ AutoPipe(x1)", horizon=3, n_experts=1, pop_size=4, use_opera=False, seed=5, export_dir=str(tmp_path))
    model.fit(weekly, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": True, "grid": False})
    assert model.ctx.adaptive_lags == [3]


def test_a_past_covariate_lag_shorter_than_the_given_horizon_is_refused(tmp_path):
    df = _series(80).assign(avail=1.0)
    model = AutoTAM("y ~ AutoPipe(x1)", horizon=7, n_experts=1, pop_size=4, export_dir=str(tmp_path))
    with pytest.raises(ValueError, match="past covariate 'avail'"):
        model.fit(df, date_col="ds", past_covariates={"avail": [3]})
