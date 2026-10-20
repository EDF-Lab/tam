# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Past-only covariates: columns known up to the forecast origin only.

Their values at the forecast rows are not known, so the covariate itself is never a feature; its lags of at least the horizon are, and the
search chooses among them. ``AutoTAM.fit(past_covariates={"availability": [24, 48]})`` builds ``availability_lag_24`` and ``availability_lag_48``
the way the lags of the target are built (per group), and offers them to the search as candidate features.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=120, groups=None):
    rng = np.random.default_rng(0)
    frames = []
    for g in (groups or [None]):
        df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "temp": rng.normal(20, 5, n),
                           "avail": rng.normal(100, 10, n)})
        df["load"] = 50 + df["temp"] + 0.3 * df["avail"].shift(7).bfill() + rng.normal(0, 1, n)
        if g is not None:
            df["site"] = g
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _model(formula="load ~ AutoPipe(temp)", **kwargs):
    return AutoTAM(formula, n_experts=1, pop_size=4, use_opera=False, **kwargs)


def _prepare(df, past_covariates, model=None, group_col=None):
    model = model or _model()
    return model.data_manager.prepare(df_train=df, date_col="ds", group_col=group_col, past_covariates=past_covariates)


def test_fit_has_a_defaulted_past_covariates_argument():
    parameter = inspect.signature(AutoTAM.fit).parameters["past_covariates"]
    assert parameter.default is None


def test_prepare_builds_the_lag_columns_of_a_past_covariate():
    df = _frame()
    ctx = _prepare(df, {"avail": [7, 14]})
    aug = ctx.df_all_aug.set_index("ds")
    expected = df.set_index("ds")["avail"]
    for lag in (7, 14):
        column = f"avail_lag_{lag}"
        assert column in aug.columns
        both = pd.concat([aug[column], expected.shift(lag)], axis=1).dropna()
        assert len(both) > 50 and np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_the_lag_columns_are_candidate_features_and_the_covariate_itself_is_not():
    ctx = _prepare(_frame(), {"avail": [7]})
    assert "avail_lag_7" in ctx.formula_config["features"]
    assert "avail" not in ctx.formula_config["features"]
    assert ctx.past_covariates == {"avail": [7]}


def test_the_lags_are_taken_inside_each_group():
    df = _frame(groups=["a", "b"])
    df.loc[df["site"] == "b", "avail"] += 1000.0
    ctx = _prepare(df, {"avail": [7]}, group_col="site")
    aug = ctx.df_all_aug  # the group labels are encoded by the pipeline: the groups are told apart by their raw values
    b = aug[aug["avail"] > 500.0]
    a = aug[aug["avail"] < 500.0]
    assert len(a) > 50 and len(b) > 50
    assert b["avail_lag_7"].dropna().min() > 500.0   # no value of group a leaks into group b
    assert a["avail_lag_7"].dropna().max() < 500.0


def test_without_past_covariates_nothing_changes():
    ctx = _prepare(_frame(), None)
    assert ctx.past_covariates == {}
    assert not [c for c in ctx.df_all_aug.columns if c.startswith("avail_lag_")]
    assert "avail_lag_7" not in ctx.formula_config["features"]


@pytest.mark.parametrize("past, fragment", [
    ({"nope": [7]}, "nope"),
    ({"avail": [0]}, "at least 1"),
    ({"avail": [7.5]}, "integer"),
    ({"load": [7]}, "target"),
])
def test_a_wrong_declaration_is_rejected_with_a_message(past, fragment):
    with pytest.raises(ValueError) as error:
        _prepare(_frame(), past)
    assert fragment in str(error.value)


def test_a_lag_shorter_than_the_horizon_is_rejected():
    """The horizon AutoTAM works with is the smallest lag of the target: a covariate lag below it would not be known at the forecast origin."""
    model = _model(lags=[24])
    with pytest.raises(ValueError) as error:
        _prepare(_frame(), {"avail": [7]}, model=model)
    message = str(error.value)
    assert "avail" in message and "7" in message and "24" in message and "horizon" in message


def test_the_past_covariate_cannot_also_be_a_feature_of_the_formula():
    model = _model("load ~ AutoPipe(temp, avail)")
    with pytest.raises(ValueError) as error:
        _prepare(_frame(), {"avail": [7]}, model=model)
    assert "avail" in str(error.value) and "lags" in str(error.value)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_past_cov"))
    try:
        df = _frame(160)
        past, future = df.iloc[:140].copy(), df.iloc[140:].copy()
        model = _model()
        model.fit(past, date_col="ds", past_covariates={"avail": [7]}, expansions=EXPANSIONS)
        yield model, past, future
    finally:
        os.chdir(previous)


def test_a_fit_with_past_covariates_predicts_the_forecast_rows_from_the_stored_history(fitted):
    model, past, future = fitted
    assert model.trained_experts
    alone = model.predict(future, date_col="ds")
    full = model.predict(pd.concat([past, future]), date_col="ds").tail(len(future))
    np.testing.assert_allclose(alone[AutoTAM.FINAL_MODEL_COLUMN].to_numpy() if AutoTAM.FINAL_MODEL_COLUMN in alone else alone.iloc[:, 0].to_numpy(),
                               full[AutoTAM.FINAL_MODEL_COLUMN].to_numpy() if AutoTAM.FINAL_MODEL_COLUMN in full else full.iloc[:, 0].to_numpy(),
                               rtol=0, atol=1e-9)
