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
