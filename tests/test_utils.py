# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.common.utils``: formula parsing, argument splitting,
feature validation, group balancing, and dummy-column injection/cleanup.
"""

import pytest
import pandas as pd

import tam
from tam.common.utils import (
    split_args_respecting_parentheses,
    parse_formula_to_terms,
    _check_features,
    _balance_groups,
    _ensure_dummies,
    _cleanup_dummies,
)


# --------------------------- argument splitting ---------------------------- #

def test_split_args_simple():
    assert split_args_respecting_parentheses("a, b, c") == ["a", "b", "c"]


def test_split_args_respects_nested_parentheses():
    # The comma inside s(x, k=5) must not split the outer argument.
    parts = split_args_respecting_parentheses("s(x, k=5), f(y, m=4)")
    assert parts == ["s(x, k=5)", "f(y, m=4)"]


def test_split_args_trailing_empty_ignored():
    assert split_args_respecting_parentheses("a, b,") == ["a", "b"]


# ----------------------------- formula parsing ----------------------------- #

def test_parse_formula_simple_terms():
    target, terms = parse_formula_to_terms("load ~ s(temp, k=10) + l(trend)")
    assert target == "load"
    assert len(terms) == 2
    assert terms[0]["type"] == "s"
    assert terms[0]["feature"] == "temp"
    assert terms[0]["params"]["k"] == 10
    assert terms[1]["type"] == "l"
    assert terms[1]["feature"] == "trend"


def test_parse_formula_typed_param_values():
    _, terms = parse_formula_to_terms("y ~ s(x, cyclic=True, basis='cubic', ratio=0.5)")
    params = terms[0]["params"]
    assert params["cyclic"] is True
    assert params["basis"] == "cubic"
    assert params["ratio"] == 0.5


def test_parse_formula_intercept_token_skipped():
    _, terms = parse_formula_to_terms("y ~ 1 + l(x)")
    assert len(terms) == 1
    assert terms[0]["feature"] == "x"


def test_parse_formula_tensor_product():
    _, terms = parse_formula_to_terms("y ~ te(s(lat), s(lon))")
    assert terms[0]["type"] == "te"
    assert terms[0]["feature"] == "interaction"


def test_parse_formula_missing_tilde_raises():
    with pytest.raises(ValueError, match="exactly one"):
        parse_formula_to_terms("load s(temp)")


def test_parse_formula_empty_rhs_raises():
    with pytest.raises(ValueError, match="No terms found"):
        parse_formula_to_terms("load ~ ")


def test_parse_formula_malformed_term_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_formula_to_terms("load ~ temp")


def test_parse_formula_term_without_arguments_raises():
    with pytest.raises(ValueError, match="no arguments"):
        parse_formula_to_terms("load ~ s()")


# --------------------------- feature validation ---------------------------- #

def test_check_features_passes_when_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    _check_features(df, ["a", "b"])  # must not raise


def test_check_features_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match="Missing required features"):
        _check_features(df, ["a", "z"])


# ----------------------------- group balancing ----------------------------- #

def _unbalanced():
    return pd.DataFrame({
        "gid": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime([
            "2022-01-01", "2022-01-02", "2022-01-03",
            "2022-01-01", "2022-01-02",
        ]),
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_balance_groups_invalid_method_raises():
    with pytest.raises(ValueError, match="'drop' or 'fill'"):
        _balance_groups(_unbalanced(), "gid", "date", method="explode")


def test_balance_groups_already_equal_is_noop():
    df = pd.DataFrame({
        "gid": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2022-01-01", "2022-01-02"] * 2),
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    mask, balanced = _balance_groups(df, "gid", "date", method="drop")
    assert bool(mask.all())
    assert len(balanced) == len(df)


def test_balance_groups_drop_truncates_to_min():
    mask, balanced = _balance_groups(_unbalanced(), "gid", "date", method="drop")
    # Both groups truncated to min count (2) -> 4 rows total.
    assert len(balanced) == 4
    assert (balanced["gid"].value_counts() == 2).all()


def test_balance_groups_fill_pads_to_max():
    mask, balanced = _balance_groups(_unbalanced(), "gid", "date", method="fill")
    # Both groups padded to max count (3) -> 6 rows total.
    assert len(balanced) == 6
    assert (balanced["gid"].value_counts() == 3).all()
    # The mask flags the single padded row as not-original.
    assert mask.sum() == 5


def test_balance_groups_empty_dataframe():
    empty = pd.DataFrame({"gid": [], "date": [], "v": []})
    mask, balanced = _balance_groups(empty, "gid", "date", method="drop")
    assert balanced.empty


# ------------------------- dummy inject / cleanup -------------------------- #

def test_ensure_and_cleanup_dummies_roundtrip():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    injected = _ensure_dummies(df, "__dummy_group__", "__dummy_date__")
    assert "__dummy_group__" in injected.columns
    assert "__dummy_date__" in injected.columns

    cleaned = _cleanup_dummies(injected, "__dummy_group__", "__dummy_date__")
    assert "__dummy_group__" not in cleaned.columns
    assert "__dummy_date__" not in cleaned.columns
    assert list(cleaned.columns) == ["v"]


def test_ensure_dummies_preserves_real_columns():
    df = pd.DataFrame({"gid": ["A"], "ts": pd.to_datetime(["2022-01-01"]), "v": [1.0]})
    out = _ensure_dummies(df, "gid", "ts")
    # Real columns are not dummy sentinels, so nothing is injected.
    assert list(out.columns) == ["gid", "ts", "v"]


def test_cleanup_dummies_keeps_real_columns():
    df = pd.DataFrame({"gid": ["A"], "ts": ["x"], "v": [1.0]})
    out = _cleanup_dummies(df, "gid", "ts")
    # Real (non-sentinel) columns must never be dropped.
    assert list(out.columns) == ["gid", "ts", "v"]
