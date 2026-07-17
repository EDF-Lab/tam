# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.parser``, the AutoTAM formula decoder.

Covers the standalone term parser and the ``FormulaParser`` macro interpreter
(AutoPipe expansion, lag injection via '@', multi-target handling, and the
target-leakage guards that strip the date column and targets from features).
"""

import pytest

from tam.model.autotam.parser import parse_formula_to_terms, FormulaParser


# ------------------------------- parse_formula_to_terms -------------------- #

def test_parse_terms_extracts_target_and_effects():
    target, terms = parse_formula_to_terms("Y ~ s(X, k=10) + l(Z)")
    assert target == "Y"
    assert {"type": "s", "feature": "X", "params": {"k": 10}} in terms
    assert {"type": "l", "feature": "Z", "params": {}} in terms


def test_parse_terms_literal_eval_typed_params():
    _, terms = parse_formula_to_terms("Y ~ s(X, k=10, basis='cubic')")
    params = terms[0]["params"]
    assert params["k"] == 10  # coerced to int by ast.literal_eval
    assert params["basis"] == "cubic"  # quotes stripped, left as string


def test_parse_terms_skips_intercept_token():
    _, terms = parse_formula_to_terms("Y ~ 1 + l(Z)")
    assert len(terms) == 1
    assert terms[0]["feature"] == "Z"


def test_parse_terms_requires_tilde():
    with pytest.raises(ValueError, match="Must contain '~'"):
        parse_formula_to_terms("Y s(X)")


# ------------------------------- FormulaParser ----------------------------- #

def test_autopipe_macro_extracts_features_and_lags():
    parser = FormulaParser()
    config = parser.parse("Load ~ AutoPipe(Temp, Humidity, Load@24)")

    assert config["targets"] == ["Load"]
    assert config["pipeline_type"] == "AutoPipe"
    assert config["features"] == ["Temp", "Humidity"]
    assert config["lags"] == {"Load_lag_24": 24}


def test_plain_rhs_without_macro_splits_on_plus():
    parser = FormulaParser()
    config = parser.parse("Y ~ a + b + c")
    assert config["features"] == ["a", "b", "c"]
    assert config["lags"] == {}


def test_multi_target_left_hand_side():
    parser = FormulaParser()
    config = parser.parse("Y1 + Y2 ~ AutoPipe(x)")
    assert config["targets"] == ["Y1", "Y2"]


def test_equals_sign_is_treated_as_target_separator():
    parser = FormulaParser()
    config = parser.parse("Y1 = Y2 ~ AutoPipe(x)")
    assert config["targets"] == ["Y1", "Y2"]


def test_date_column_excluded_from_features():
    parser = FormulaParser()
    config = parser.parse("Y ~ AutoPipe(Temp, ds)", date_col="ds")
    assert "ds" not in config["features"]


def test_target_excluded_from_features():
    parser = FormulaParser()
    config = parser.parse("Load ~ AutoPipe(Temp, Load)")
    assert "Load" not in config["features"]


def test_static_pipeline_with_lags_warns(capsys):
    parser = FormulaParser()
    parser.parse("Load ~ StaticTAM(Temp, Load@7)")
    captured = capsys.readouterr()
    assert "Lags detected" in captured.out


def test_parse_requires_tilde():
    parser = FormulaParser()
    with pytest.raises(ValueError, match="Invalid formula syntax"):
        parser.parse("Load AutoPipe(Temp)")
