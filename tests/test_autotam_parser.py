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

from tam.model.autotam.parser import (
    parse_formula_to_terms,
    FormulaParser,
    canonicalize_term,
    terms_are_equivalent,
    canonicalize_formula,
)
from tam.model.autotam import (
    canonicalize_term as pkg_canonicalize_term,
    terms_are_equivalent as pkg_terms_are_equivalent,
    canonicalize_formula as pkg_canonicalize_formula,
)


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


# --------------------- Term Canonicalization & Equivalence ----------------- #

def test_package_level_exports():
    assert pkg_canonicalize_term is canonicalize_term
    assert pkg_terms_are_equivalent is terms_are_equivalent
    assert pkg_canonicalize_formula is canonicalize_formula


def test_canonicalize_marginal_term_sorts_parameters_alphabetically():
    # Parameters provided in reverse/different orders
    term1 = "c(WeekDays, topo='nominal', n_cat=7)"
    term2 = "c(WeekDays, n_cat=7, topo='nominal')"
    
    assert canonicalize_term(term1) == "c(WeekDays, n_cat=7, topo='nominal')"
    assert canonicalize_term(term2) == "c(WeekDays, n_cat=7, topo='nominal')"
    assert canonicalize_term(term1) == canonicalize_term(term2)


def test_canonicalize_marginal_term_normalizes_quotes_and_literals():
    # Single quotes, double quotes, unquoted strings, numbers, booleans
    t_single = "c(X, topo='nominal', n_cat=7, flag=True)"
    t_double = 'c(X, topo="nominal", n_cat=7, flag=True)'
    t_unquoted = "c(X, topo=nominal, n_cat=7, flag=True)"
    
    expected = "c(X, flag=True, n_cat=7, topo='nominal')"
    assert canonicalize_term(t_single) == expected
    assert canonicalize_term(t_double) == expected
    assert canonicalize_term(t_unquoted) == expected


def test_canonicalize_marginal_term_without_params():
    assert canonicalize_term("s(X)") == "s(X)"
    assert canonicalize_term("l(Z)") == "l(Z)"
    assert canonicalize_term("  s(  X  )  ") == "s(X)"


def test_canonicalize_marginal_term_with_ast_dict():
    ast_dict = {"type": "c", "feature": "WeekDays", "params": {"topo": "nominal", "n_cat": 7}}
    assert canonicalize_term(ast_dict) == "c(WeekDays, n_cat=7, topo='nominal')"

    ast_dict_no_params = {"type": "l", "feature": "Z", "params": {}}
    assert canonicalize_term(ast_dict_no_params) == "l(Z)"


def test_canonicalize_intercept_and_bare_tokens():
    assert canonicalize_term("1") == "1"
    assert canonicalize_term("  1  ") == "1"


def test_canonicalize_tensor_product_sorts_subterms_and_kwargs():
    # Sub-terms permuted and params inside sub-terms permuted
    te1 = "te(s(x, k=5), c(y, topo='nominal', n_cat=7))"
    te2 = "te(c(y, n_cat=7, topo='nominal'), s(x, k=5))"
    
    expected = "te(c(y, n_cat=7, topo='nominal'), s(x, k=5))"
    assert canonicalize_term(te1) == expected
    assert canonicalize_term(te2) == expected
    assert canonicalize_term(te1) == canonicalize_term(te2)


def test_canonicalize_tensor_product_with_kwargs():
    te_kw1 = "te(s(x, k=5), c(y), ap=-30)"
    te_kw2 = "te(ap=-30, c(y), s(x, k=5))"
    expected = "te(c(y), s(x, k=5), ap=-30)"
    assert canonicalize_term(te_kw1) == expected
    assert canonicalize_term(te_kw2) == expected


def test_canonicalize_tensor_product_with_tuple_kwargs():
    te_kw1 = "te(s(x, k=5), c(y), bs=('cr', 'ps'))"
    te_kw2 = "te(bs=('cr', 'ps'), c(y), s(x, k=5))"
    expected = "te(c(y), s(x, k=5), bs=('cr', 'ps'))"
    assert canonicalize_term(te_kw1) == expected
    assert canonicalize_term(te_kw2) == expected


def test_canonicalize_tensor_product_ast_dict():
    ast_te = {
        "type": "te",
        "params": {
            "s(x, k=5)": None,
            "c(y, topo='nominal', n_cat=7)": None,
            "__sub_y_n_cat_0_0": 7,
        }
    }
    assert canonicalize_term(ast_te) == "te(c(y, n_cat=7, topo='nominal'), s(x, k=5))"


def test_terms_are_equivalent():
    assert terms_are_equivalent(
        "c(WeekDays, topo='nominal', n_cat=7)",
        "c(WeekDays, n_cat=7, topo='nominal')"
    )
    assert terms_are_equivalent(
        {"type": "c", "feature": "WeekDays", "params": {"n_cat": 7, "topo": "nominal"}},
        "c(WeekDays, topo='nominal', n_cat=7)"
    )
    assert terms_are_equivalent(
        "te(s(x, k=5), c(y, topo='nominal', n_cat=7))",
        "te(c(y, n_cat=7, topo='nominal'), s(x, k=5))"
    )
    # Non-equivalent cases
    assert not terms_are_equivalent("s(X, k=5)", "s(X, k=10)")
    assert not terms_are_equivalent("s(X)", "s(Y)")
    assert not terms_are_equivalent("s(X)", "l(X)")


def test_canonicalize_formula():
    formula = "Load ~ s(Temp, k=10) + c(Day, topo='nominal', n_cat=7) + s(Temp, k=10) + c(Day, n_cat=7, topo='nominal')"
    expected = "Load ~ c(Day, n_cat=7, topo='nominal') + s(Temp, k=10)"
    assert canonicalize_formula(formula) == expected


def test_canonicalize_formula_with_tensors_and_intercept():
    formula = "Y ~ te(s(x, k=5), c(y)) + 1 + te(c(y), s(x, k=5)) + l(z)"
    expected = "Y ~ l(z) + te(c(y), s(x, k=5))"
    assert canonicalize_formula(formula) == expected


def test_canonicalize_formula_pure_intercept():
    assert canonicalize_formula("Y ~ 1") == "Y ~ 1"


def test_canonicalize_formula_multi_target():
    formula = "Y1 + Y2 ~ s(b) + s(a)"
    expected = "Y1 + Y2 ~ s(a) + s(b)"
    assert canonicalize_formula(formula) == expected


def test_canonicalize_formula_requires_tilde():
    with pytest.raises(ValueError, match="Must contain '~'"):
        canonicalize_formula("Y s(X)")

