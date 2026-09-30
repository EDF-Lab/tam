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


def test_autotam_init_extracts_and_propagates_mandatory_terms():
    at = AutoTAM("load ~ s(temp, k=10) + AutoPipe(temp, humidity)")
    assert at.mandatory_terms == ["s(temp, k=10)"]
    assert at.data_manager.mandatory_terms == ["s(temp, k=10)"]
    assert at.discoverer.mandatory_terms == ["s(temp, k=10)"]


def test_autotam_init_rejects_mandatory_terms_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument 'mandatory_terms'"):
        AutoTAM("load ~ AutoPipe(temp, humidity)", mandatory_terms=["s(temp, k=10)"])


def test_autotam_init_validates_mandatory_terms_syntactically():
    with pytest.raises(ValueError, match=r"Unbalanced parentheses|Invalid mandatory term"):
        AutoTAM("load ~ invalid_syntax_term((( + AutoPipe(temp, humidity)")
    with pytest.raises(ValueError, match=r"Invalid mandatory term"):
        AutoTAM("load ~ bare_var + AutoPipe(temp, humidity)")


def test_autotam_fit_validates_mandatory_terms_semantically():
    df = _frame()
    at = AutoTAM("load ~ s(unknown_col, k=5) + AutoPipe(temp, humidity)")
    with pytest.raises(ValueError, match="unknown_col"):
        at.fit(df, date_col="ds")


def test_expert_predictions_raises_on_missing_external_mandatory_features():
    df = _frame(60)
    at = AutoTAM(
        "load ~ s(humidity, k=5) + AutoPipe(temp)",
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
        "load ~ s(humidity, k=5) + AutoPipe(temp)",
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
        "load ~ s(humidity, k=5) + AutoPipe(temp)",
        max_generations=1,
        population_size=2,
    )
    at.fit(df, date_col="ds", refit_on_full_train=True)
    df_test = df.iloc[-10:].copy()
    preds = at.predict(df_test)
    assert len(preds) == 10


def test_autotam_with_mandatory_terms_and_mandatory_variables():
    df = _frame(60)
    at = AutoTAM(
        "load ~ s(temp, k=10) + AutoPipe(temp, humidity)",
        mandatory_variables=["humidity"],
        pop_size=6,
    )
    at.fit(df, date_col="ds")

    # Verify that formulas evaluated during search and expert expansion have balanced parentheses
    assert len(at.chronological_test_log) > 0
    for entry in at.chronological_test_log:
        formula = entry.get("Formula", "")
        assert formula.count("(") == formula.count(")"), f"Unbalanced parentheses in {formula}"
        if entry.get("Model_Type", "").startswith("StaticTAM"):
            # All base models must subsume the mandatory term s(temp, k=10) and cover mandatory variable humidity
            from tam.model.autotam.parser import term_subsumes
            _, terms = parse_formula_to_terms(formula)
            assert any(term_subsumes(t, "s(temp, k=10)") for t in terms), f"Missing mandatory term in {formula}"
            assert any(t.get("feature") == "humidity" for t in terms), f"Missing mandatory var in {formula}"

    df_test = df.iloc[-10:].copy()
    preds = at.predict(df_test)
    assert len(preds) == 10
    assert not preds.empty


