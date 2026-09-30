# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Tests for AutoTAM with mandatory terms and mandatory variables.
Verifies that formula generation produces well-formed formulas without parenthesis mismatch.
"""

import pytest
import numpy as np
import pandas as pd
from tam.model.autotam.auto_tam import AutoTAM


def _make_df(n=60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
        "humidity": rng.uniform(30, 90, n),
    })


def test_only_mandatory_terms():
    df = _make_df()
    at = AutoTAM(
        "load ~ s(temp, k=10) + AutoPipe(temp, humidity)",
        pop_size=6,
    )
    at.fit(df, date_col="ds")
    assert len(at.trained_experts) > 0
    for entry in at.chronological_test_log:
        f = entry.get("Formula", "")
        assert f.count("(") == f.count(")"), f"Unbalanced parentheses in {f}"


def test_only_mandatory_variables():
    df = _make_df()
    at = AutoTAM(
        "load ~ AutoPipe(temp, humidity)",
        mandatory_variables=["humidity"],
        pop_size=6,
    )
    at.fit(df, date_col="ds")
    assert len(at.trained_experts) > 0
    for entry in at.chronological_test_log:
        f = entry.get("Formula", "")
        assert f.count("(") == f.count(")"), f"Unbalanced parentheses in {f}"
        if entry.get("Model_Type", "").startswith("StaticTAM"):
            assert "humidity" in f


def test_both_mandatory_terms_and_variables():
    df = _make_df()
    at = AutoTAM(
        "load ~ s(temp, k=10) + AutoPipe(temp, humidity)",
        mandatory_variables=["humidity"],
        pop_size=6,
    )
    at.fit(df, date_col="ds")
    assert len(at.trained_experts) > 0
    from tam.model.autotam.parser import term_subsumes, parse_formula_to_terms
    for entry in at.chronological_test_log:
        f = entry.get("Formula", "")
        assert f.count("(") == f.count(")"), f"Unbalanced parentheses in {f}"
        if entry.get("Model_Type", "").startswith("StaticTAM"):
            _, terms = parse_formula_to_terms(f)
            assert any(term_subsumes(t, "s(temp, k=10)") for t in terms), f"Missing mandatory term in {f}"
            assert any(t.get("feature") == "humidity" for t in terms), f"Missing mandatory var in {f}"
