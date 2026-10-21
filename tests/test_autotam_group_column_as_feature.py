# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
A group column that is also named in the formula (``group_col='hour'`` with ``hour`` among the features).

A model per hour is a common design for load forecasting, so a user will write the hour both as the key of the groups
and in the formula. The search takes it as the key: it fits one model per group, the formulas keep the other features,
and ``fit()`` and ``predict()`` return one forecast per row, never a ``KeyError``.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from tam.model.autotam.auto_tam import AutoTAM


def _hourly(n_days=9):
    rng = np.random.default_rng(0)
    ds = pd.date_range("2023-01-01", periods=24 * n_days, freq="h")
    df = pd.DataFrame({"ds": ds, "hour": ds.hour, "temp": rng.normal(15, 5, len(ds))})
    df["load"] = 50 + df["temp"] + 8 * np.sin(df["hour"] / 24 * 2 * np.pi) + rng.normal(0, 1, len(ds))
    return df


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_group_feature"))
    try:
        df = _hourly()
        train, test = df.iloc[:24 * 7].copy(), df.iloc[24 * 7:].copy()
        model = AutoTAM("load ~ AutoPipe(temp, hour)", n_experts=1, pop_size=4, use_opera=False)
        model.fit(train, date_col="ds", group_col="hour",
                  expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
        yield model, train, test
    finally:
        os.chdir(previous)


def test_fit_with_the_group_column_in_the_formula_keeps_experts(fitted):
    model, _, _ = fitted
    assert model.trained_experts


def test_predict_returns_one_forecast_per_row(fitted):
    model, _, test = fitted
    out = model.predict(test, date_col="ds")
    assert len(out) == len(test)
    assert out.shape[1] >= 1
    assert out.notna().any().any()


def test_the_group_column_is_the_key_and_not_a_feature_of_the_models(fitted):
    """Inside one hour the hour is constant: a term on it carries no information, the search must not build one."""
    model, _, _ = fitted
    formulas = [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]
    assert formulas
    assert not [f for f in formulas if re.search(r"\(\s*hour\b", f)], "a term on the group column was built"


def test_a_group_column_that_is_not_a_feature_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _hourly(n_days=10)
    df["site"] = np.where(np.arange(len(df)) % 2 == 0, "a", "b")
    model = AutoTAM("load ~ AutoPipe(temp)", n_experts=1, pop_size=4, use_opera=False)
    model.fit(df.iloc[:24 * 8].copy(), date_col="ds", group_col="site",
              expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    out = model.predict(df.iloc[24 * 8:].copy(), date_col="ds")
    assert len(out) == len(df.iloc[24 * 8:])
