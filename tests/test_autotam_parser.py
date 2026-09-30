# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Amaury Durand
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
    term_subsumes,
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


def test_plain_rhs_without_macro_raises_error():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"must contain exactly one pipeline macro.*None found"):
        parser.parse("Y ~ a + b + c")


def test_formula_parser_multiple_macros_raises_error():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"must contain exactly one pipeline macro, but found 2"):
        parser.parse("Y ~ AutoPipe(a, b) + AdaptTAM(c)")


def test_formula_parser_empty_macro_raises_error():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"Pipeline macro 'AutoPipe' cannot be empty"):
        parser.parse("Y ~ AutoPipe()")
    with pytest.raises(ValueError, match=r"Pipeline macro 'AdaptTAM' cannot be empty"):
        parser.parse("Y ~ AdaptTAM()")


def test_formula_parser_macro_position_invariance():
    parser = FormulaParser()

    # Trailing macro
    c1 = parser.parse("Y ~ s(x, k=10) + c(y, n_cat=2) + AutoPipe(x, z)")
    assert c1["pipeline_type"] == "AutoPipe"
    assert c1["features"] == ["x", "z"]
    assert c1["mandatory_terms"] == ["s(x, k=10)", "c(y, n_cat=2)"]

    # Leading macro
    c2 = parser.parse("Y ~ AutoPipe(x, z) + s(x, k=10) + c(y, n_cat=2)")
    assert c2["pipeline_type"] == "AutoPipe"
    assert c2["features"] == ["x", "z"]
    assert c2["mandatory_terms"] == ["s(x, k=10)", "c(y, n_cat=2)"]

    # Middle macro
    c3 = parser.parse("Y ~ s(x, k=10) + AutoPipe(x, z) + c(y, n_cat=2)")
    assert c3["pipeline_type"] == "AutoPipe"
    assert c3["features"] == ["x", "z"]
    assert c3["mandatory_terms"] == ["s(x, k=10)", "c(y, n_cat=2)"]


def test_formula_parser_no_mandatory_terms():
    parser = FormulaParser()
    config = parser.parse("Y ~ AutoPipe(Temp, Humidity)")
    assert config["mandatory_terms"] == []
    assert config["features"] == ["Temp", "Humidity"]


def test_formula_parser_rejects_explicit_intercept():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"Explicit intercept '1' is not permitted as a mandatory term"):
        parser.parse("Y ~ 1 + AutoPipe(x)")
    with pytest.raises(ValueError, match=r"Explicit intercept '1' is not permitted as a mandatory term"):
        parser.parse("Y ~ AutoPipe(x) + 1")


def test_formula_parser_rejects_bare_identifiers():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"Invalid mandatory term 'x': Bare identifiers without an effect wrapper are not permitted"):
        parser.parse("Y ~ x + AutoPipe(z)")
    with pytest.raises(ValueError, match=r"Invalid mandatory term 'x': Bare identifiers without an effect wrapper are not permitted"):
        parser.parse("Y ~ AutoPipe(z) + x")


def test_formula_parser_rejects_bare_macro_identifier():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"must contain exactly one pipeline macro.*None found"):
        parser.parse("Y ~ AutoPipe")
    with pytest.raises(ValueError, match=r"Invalid mandatory term 'AutoPipe': Bare identifiers"):
        parser.parse("Y ~ AutoPipe + AutoPipe(x)")


def test_formula_parser_supported_effects_whitelist():
    parser = FormulaParser()
    # All 13 supported effect types: s, c, l, te, f, p, rbf, w, n, phys, pid, t, lt
    formula = (
        "Y ~ s(x1) + c(x2) + l(x3) + te(s(x1), c(x2)) + f(x4) + p(x5) + "
        "rbf(x6) + w(x7) + n(x8) + phys(x9) + pid(x10) + t(x11) + lt(x12) + "
        "AutoPipe(x1, x2, x3)"
    )
    config = parser.parse(formula)
    assert len(config["mandatory_terms"]) == 13
    assert config["mandatory_terms"][0] == "s(x1)"
    assert config["mandatory_terms"][3] == "te(s(x1), c(x2))"
    assert config["mandatory_terms"][-1] == "lt(x12)"


def test_formula_parser_rejects_unknown_effect():
    parser = FormulaParser()
    with pytest.raises(ValueError, match=r"Unknown effect basis function 'unknown_func' in term 'unknown_func\(x\)'"):
        parser.parse("Y ~ unknown_func(x) + AutoPipe(z)")


def test_formula_parser_canonical_deduplication():
    parser = FormulaParser()
    # Exact duplicate
    with pytest.raises(ValueError, match=r"Duplicate mandatory term detected in formula: 's\(x, k=10\)'"):
        parser.parse("Y ~ s(x, k=10) + s(x, k=10) + AutoPipe(z)")

    # Parameter permutation duplicate
    with pytest.raises(ValueError, match=r"Duplicate mandatory term detected in formula: 'c\(x, topo='nominal', n_cat=7\)'"):
        parser.parse("Y ~ c(x, n_cat=7, topo='nominal') + c(x, topo='nominal', n_cat=7) + AutoPipe(z)")

    # Tensor sub-term permutation duplicate
    with pytest.raises(ValueError, match=r"Duplicate mandatory term detected in formula: 'te\(c\(y\), s\(x, k=5\)\)'"):
        parser.parse("Y ~ te(s(x, k=5), c(y)) + te(c(y), s(x, k=5)) + AutoPipe(z)")


def test_formula_parser_preserves_verbatim_terms_and_order():
    parser = FormulaParser()
    formula = "Y ~ s(  Temp,   k=10  ) + c(Day, n_cat=7) + AutoPipe(Temp, Humidity)"
    config = parser.parse(formula)
    assert config["mandatory_terms"] == ["s(  Temp,   k=10  )", "c(Day, n_cat=7)"]


def test_formula_parser_delimiter_integrity():
    parser = FormulaParser()
    with pytest.raises(ValueError, match="leading delimiter"):
        parser.parse("Y ~ + AutoPipe(x)")
    with pytest.raises(ValueError, match="trailing delimiter"):
        parser.parse("Y ~ AutoPipe(x) +")
    with pytest.raises(ValueError, match="consecutive delimiters"):
        parser.parse("Y ~ AutoPipe(x) ++ s(y)")
    with pytest.raises(ValueError, match="unclosed opening parenthesis"):
        parser.parse("Y ~ AutoPipe(x) + s(y, k=10")


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


def test_canonicalize_term_multi_term_string_does_not_raise_unbalanced_parentheses():
    res = canonicalize_term("s(x, k=10) + l(y)")
    assert res == "l(y) + s(x, k=10)"
    res2 = canonicalize_term("l(b) + s(a)")
    assert res2 == "l(b) + s(a)"


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


def test_canonicalize_formula_delimiter_integrity_errors():
    with pytest.raises(ValueError, match="leading delimiter"):
        canonicalize_formula("Y ~ + s(X)")
    with pytest.raises(ValueError, match="trailing delimiter"):
        canonicalize_formula("Y ~ s(X) +")
    with pytest.raises(ValueError, match="consecutive delimiters"):
        canonicalize_formula("Y ~ s(X) + + s(Y)")


def test_canonicalize_formula_unbalanced_parentheses():
    with pytest.raises(ValueError, match="unclosed opening parenthesis"):
        canonicalize_formula("Y ~ s(X, k=5")
    with pytest.raises(ValueError, match="unexpected closing parenthesis"):
        canonicalize_formula("Y ~ s(X)) + s(Y)")


# --------------------- Term Subsumption Matching --------------------------- #

def test_term_subsumes_marginal_exact_and_tuned_free_parameters():
    # Exactly equivalent terms subsume each other
    assert term_subsumes("s(x, k=10)", "s(x, k=10)")
    assert term_subsumes("c(WeekDays, n_cat=7, topo='nominal')", "c(WeekDays, topo='nominal', n_cat=7)")

    # Candidate with tuned free parameters subsumes sparser mandatory specification
    assert term_subsumes("s(x, k=10, m=2)", "s(x, k=10)")
    assert term_subsumes("s(x, k=10, basis='cubic', ratio=0.5)", "s(x, k=10)")
    assert term_subsumes("s(x, k=10)", "s(x)")
    assert term_subsumes("l(x)", "l(x)")

    # Mismatched parameters do NOT subsume
    assert not term_subsumes("s(x, k=5)", "s(x, k=10)")
    assert not term_subsumes("s(x, k=10)", "s(x, k=10, m=2)")  # Mandatory has m=2, candidate lacks it
    assert not term_subsumes("s(x, k=10, m=1)", "s(x, k=10, m=2)")

    # Mismatched feature or effect type do NOT subsume
    assert not term_subsumes("s(y, k=10)", "s(x, k=10)")
    assert not term_subsumes("f(x, k=10)", "s(x, k=10)")


def test_term_subsumes_ast_dict_and_string_mix():
    ast_candidate = {"type": "s", "feature": "temp", "params": {"k": 10, "m": 2}}
    assert term_subsumes(ast_candidate, "s(temp, k=10)")
    assert term_subsumes("s(temp, k=10, m=2)", {"type": "s", "feature": "temp", "params": {"k": 10}})
    assert not term_subsumes(ast_candidate, "s(temp, k=15)")


def test_term_subsumes_tensor_interaction_invariance_and_free_parameters():
    # Identical sub-terms
    assert term_subsumes("te(s(x1, k=5), s(x2, k=10))", "te(s(x1, k=5), s(x2, k=10))")

    # Permuted sub-terms
    assert term_subsumes("te(s(x2, k=10), s(x1, k=5))", "te(s(x1, k=5), s(x2, k=10))")

    # Permuted sub-terms with tuned free parameters in sub-terms
    assert term_subsumes("te(s(x2, k=10, m=2), s(x1, k=5, basis='cubic'))", "te(s(x1, k=5), s(x2, k=10))")

    # Sub-terms matching when mandatory has unconstrained sub-terms
    assert term_subsumes("te(s(x1, k=5), s(x2, k=10))", "te(s(x1), s(x2))")
    assert term_subsumes("te(s(x2, k=10), s(x1, k=5))", "te(s(x1), s(x2))")

    # Top-level kwargs matching
    assert term_subsumes("te(s(x1), s(x2), bs='cr')", "te(s(x1), s(x2), bs='cr')")
    assert term_subsumes("te(s(x2), s(x1), bs='cr', ap=-30)", "te(s(x1), s(x2), bs='cr')")

    # Mismatched top-level kwarg
    assert not term_subsumes("te(s(x1), s(x2), bs='ps')", "te(s(x1), s(x2), bs='cr')")

    # Mismatched sub-term parameters
    assert not term_subsumes("te(s(x1, k=8), s(x2, k=10))", "te(s(x1, k=5), s(x2, k=10))")

    # Mismatched participating features
    assert not term_subsumes("te(s(x1), s(x3))", "te(s(x1), s(x2))")


