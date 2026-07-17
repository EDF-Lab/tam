# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.pipeline.context.PipelineContext``, the
shared pipeline state holder and its complexity-estimation / score-penalty maths.
"""

import pytest

from tam.model.autotam.pipeline.context import PipelineContext


def test_context_defaults():
    ctx = PipelineContext()
    assert ctx.target == ""
    assert ctx.optimization_metric == "rmse"
    assert ctx.complexity_penalty == 1.0
    assert ctx.lags == [] and ctx.cv_folds == []


@pytest.mark.parametrize("formula,expected_k", [
    ("y ~ l(x)", 1),                              # linear -> 1 coefficient
    ("y ~ s(x, k=10)", 13),                       # knots(10) + degree(3)
    ("y ~ s(x, k=10, deg=2)", 12),                # knots(10) + degree(2)
    ("y ~ f(x, k=5)", 10),                        # 2 * harmonics
    ("y ~ p(x, deg=8)", 8),                       # polynomial degree
    ("y ~ rbf(x, n_centers=20)", 20),             # centers
    ("y ~ t(x, n_trees=10, max_depth=3)", 80),    # trees * 2^depth
    ("y ~ w(x, n_scales=3, n_locations=10)", 30), # scales * locations
    ("y ~ n(x, n_neurons=10, n_hidden_layers=1)", 50),  # neurons * layers * 5
    ("y ~ c(x, n_cat=7)", 6),                     # n_cat - 1
    ("y ~ te(s(x, k=5), s(z, k=4))", 56),         # (5+3) * (4+3)
    ("y ~ s(x, k=10) + l(z)", 14),                # additive sum
])
def test_estimate_complexity(formula, expected_k):
    ctx = PipelineContext()
    assert ctx.estimate_complexity(formula) == expected_k


def test_estimate_complexity_invalid_formula_returns_one():
    ctx = PipelineContext()
    assert ctx.estimate_complexity("no tilde here") == 1
    assert ctx.estimate_complexity(None) == 1


def test_penalize_score_adds_complexity_cost():
    ctx = PipelineContext()
    ctx.complexity_penalty = 1.0
    base = 10.0
    penalized = ctx.penalize_score(base, "y ~ l(x)", n_samples=100)
    # k=1, n=100 -> score * (1 + 1*(1/100))
    assert penalized == pytest.approx(10.0 * 1.01)


def test_penalize_score_infinite_inputs_return_inf():
    ctx = PipelineContext()
    assert ctx.penalize_score(float("inf"), "y ~ l(x)", 100) == float("inf")
    assert ctx.penalize_score(10.0, "y ~ l(x)", 0) == float("inf")


def test_penalize_score_heavier_formula_costs_more():
    ctx = PipelineContext()
    light = ctx.penalize_score(10.0, "y ~ l(x)", n_samples=100)
    heavy = ctx.penalize_score(10.0, "y ~ t(x, n_trees=50, max_depth=6)", n_samples=100)
    assert heavy > light
