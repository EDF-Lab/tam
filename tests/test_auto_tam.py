# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Unit and integration tests for AutoTAM mandatory_terms search constraint.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.common.utils import parse_formula_to_terms


def _frame(n=100):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
        "humidity": rng.uniform(30, 90, n),
    })


def test_autotam_init_normalizes_and_propagates_mandatory_terms():
    # Passed as list
    at1 = AutoTAM("load ~ AutoPipe(temp, humidity)", mandatory_terms=["s(temp, k=10)"])
    assert at1.mandatory_terms == ["s(temp, k=10)"]
    assert at1.data_manager.mandatory_terms == ["s(temp, k=10)"]
    assert at1.discoverer.mandatory_terms == ["s(temp, k=10)"]

    # Passed as single string
    at2 = AutoTAM("load ~ AutoPipe(temp, humidity)", mandatory_terms="s(temp, k=10)")
    assert at2.mandatory_terms == ["s(temp, k=10)"]
    assert at2.data_manager.mandatory_terms == ["s(temp, k=10)"]
    assert at2.discoverer.mandatory_terms == ["s(temp, k=10)"]


def test_autotam_fit_validates_mandatory_terms_syntactically():
    df = _frame()
    at = AutoTAM("load ~ AutoPipe(temp, humidity)", mandatory_terms=["invalid_syntax_term((("])
    with pytest.raises(ValueError, match="Invalid mandatory term"):
        at.fit(df, date_col="ds")


def test_autotam_fit_validates_mandatory_terms_semantically():
    df = _frame()
    at = AutoTAM("load ~ AutoPipe(temp, humidity)", mandatory_terms=["s(unknown_col, k=5)"])
    with pytest.raises(ValueError, match="unknown_col"):
        at.fit(df, date_col="ds")


def test_expert_predictions_raises_on_missing_external_mandatory_features():
    df = _frame(60)
    at = AutoTAM(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["s(humidity, k=5)"],
        max_generations=1,
        population_size=2,
    )
    at.fit(df, date_col="ds")

    # Missing external mandatory column 'humidity'
    df_test_missing = df[["ds", "load", "temp"]].iloc[:10].copy()
    with pytest.raises(ValueError, match="external mandatory"):
        at.predict(df_test_missing)


def test_autotam_end_to_end_fit_and_predict_with_external_mandatory_terms():
    df = _frame(60)
    at = AutoTAM(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["s(humidity, k=5)"],
        max_generations=1,
        population_size=2,
    )
    at.fit(df, date_col="ds")

    # Verify all evaluated base GAM formulas only have the specified mandatory term for humidity
    for entry in at.chronological_test_log:
        if entry.get("Model_Type", "").startswith("StaticTAM"):
            formula = entry.get("Formula", "")
            _, terms = parse_formula_to_terms(formula)
            for t in terms:
                if t.get("feature") == "humidity":
                    assert t.get("type") == "s"

    # Verify all final static models in trained experts only have s(humidity)
    for exp in at.trained_experts:
        if exp["type"] == "static":
            f = getattr(exp["model"], "formula_", "")
            _, terms = parse_formula_to_terms(f)
            for t in terms:
                if t.get("feature") == "humidity":
                    assert t.get("type") == "s"

    # Test predict succeeds on test set containing humidity
    df_test = df.iloc[-10:].copy()
    preds = at.predict(df_test)
    assert len(preds) == 10
    assert not preds.empty
    assert preds.notna().any().any()

    # Test predict_online succeeds
    preds_online = at.predict_online(df_test)
    assert len(preds_online) == 10


def test_autotam_end_to_end_full_refit_with_external_mandatory_terms():
    df = _frame(60)
    at = AutoTAM(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["s(humidity, k=5)"],
        max_generations=1,
        population_size=2,
    )
    at.fit(df, date_col="ds", refit_on_full_train=True)
    df_test = df.iloc[-10:].copy()
    preds = at.predict(df_test)
    assert len(preds) == 10

