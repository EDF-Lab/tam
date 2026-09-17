# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for the parsed-term complexity estimation of
``tam.model.autotam.pipeline.context`` (``term_complexity``, ``estimate_complexity``).
"""

import pytest

from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.pipeline.context import PipelineContext, term_complexity


def _terms(formula):
    return parse_formula_to_terms(formula)[1]


@pytest.mark.parametrize("formula", [
    "y ~ f(toy, m=22, s=2, ap=-3.0)",
    "y ~ c(day_type_week, n_cat=7, topo='nominal', p_order=1) + l(Load_d1, ap=-30.0)",
    "y ~ te(f(toy, m=6, s=2), c(day_type_week, n_cat=7, topo='nominal'))",
    "y ~ w(temperature, n_scales=4, n_locations=13, ap=-5.0) + s(temperature, k=24, deg=3, p=2)",
])
def test_string_and_parsed_terms_agree(formula):
    ctx = PipelineContext()
    assert ctx.estimate_complexity(formula) == ctx.estimate_complexity(_terms(formula))


def test_fourier_reads_m_and_accepts_legacy_k():
    assert term_complexity(_terms("y ~ f(x, m=20)")[0]) == 40
    assert term_complexity(_terms("y ~ f(x, k=5)")[0]) == 10
    assert term_complexity(_terms("y ~ f(x)")[0]) == 20


def test_tensor_surrogate_never_exceeds_the_true_width():
    # f(m=6) has 12 columns and a two-level categorical 1: width 12, surrogate 12 + 1 + sqrt(12) > 12.
    assert term_complexity(_terms("y ~ te(f(x, m=6), c(z, n_cat=2))")[0]) == 12


def test_grid_tokens_fall_back_to_defaults():
    assert PipelineContext().estimate_complexity("y ~ s(x, k='grid_k_x_s', deg=3)") == 13


def test_combined_dynamic_formula_includes_the_static_base():
    ctx = PipelineContext()
    kalman = "Load ~ effect_a + effect_b - 1 + Load ~ s(x, k=10) + l(z)"
    adaptive = "ResidualLoad ~ l(ResidualLoad_lag_1) + l(effect_a) + Load ~ c(d, n_cat=7)"
    assert ctx.estimate_complexity(kalman) == 1 + 1 + 13 + 1
    assert ctx.estimate_complexity(adaptive) == 1 + 1 + 6


def test_unparsable_terms_cost_one_column_each():
    assert PipelineContext().estimate_complexity("y ~ l(x) + ???(z") == 2


def test_tree_leaves_follow_the_architecture():
    assert term_complexity(_terms("y ~ t(x, n_trees=10, max_depth=2)")[0]) == 40
    assert term_complexity(_terms("y ~ t(x, n_trees=10, max_leaves=50)")[0]) == 500


def test_linear_tree_and_pid_widths():
    assert term_complexity(_terms("y ~ lt(x, max_depth=3)")[0]) == 16
    assert term_complexity(_terms("y ~ lt(x, max_leaves=50)")[0]) == 100
    assert term_complexity(_terms("y ~ pid(Load_d1, w=7, d_pen=10)")[0]) == 3
