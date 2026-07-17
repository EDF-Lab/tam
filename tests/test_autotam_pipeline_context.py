# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.pipeline.context.PipelineContext``.

Covers the complexity estimation that drives the Information-Criterion penalty
and the penalize_score helper. Both are pure functions with no I/O.
"""

import pytest
from tam.model.autotam.pipeline.context import PipelineContext


def _ctx(penalty: float = 1.0) -> PipelineContext:
    return PipelineContext(complexity_penalty=penalty)


# --------------------------------------------------------------------------- #
# estimate_complexity
# --------------------------------------------------------------------------- #

def test_estimate_complexity_linear_term():
    ctx = _ctx()
    assert ctx.estimate_complexity("y ~ l(x)") == 1


def test_estimate_complexity_spline_uses_knots_plus_degree():
    ctx = _ctx()
    # s(x, k=10, deg=3) -> 10 + 3 = 13
    assert ctx.estimate_complexity("y ~ s(x, k=10, deg=3)") == 13


def test_estimate_complexity_fourier_uses_two_per_harmonic():
    ctx = _ctx()
    # f(x, k=5) -> 5*2 = 10
    assert ctx.estimate_complexity("y ~ f(x, k=5)") == 10


def test_estimate_complexity_chebyshev_uses_degree():
    ctx = _ctx()
    assert ctx.estimate_complexity("y ~ p(x, deg=8)") == 8


def test_estimate_complexity_rbf_uses_n_centers():
    ctx = _ctx()
    assert ctx.estimate_complexity("y ~ rbf(x, n_centers=20)") == 20


def test_estimate_complexity_tree_uses_leaves():
    ctx = _ctx()
    # t(x, n_trees=10, max_depth=2) -> 10 * 2^2 = 40
    assert ctx.estimate_complexity("y ~ t(x, n_trees=10, max_depth=2)") == 40


def test_estimate_complexity_wavelet_uses_scales_times_locs():
    ctx = _ctx()
    # w(x, n_scales=3, n_locations=10) -> 30
    assert ctx.estimate_complexity("y ~ w(x, n_scales=3, n_locations=10)") == 30


def test_estimate_complexity_neural_heavy_penalty():
    ctx = _ctx()
    # n(x, n_neurons=10, n_hidden_layers=2) -> 10*2*5 = 100
    assert ctx.estimate_complexity("y ~ n(x, n_neurons=10, n_hidden_layers=2)") == 100


def test_estimate_complexity_tensor_product_multiplies():
    ctx = _ctx()
    # te(s(a, k=5, deg=3), s(b, k=4, deg=3)) -> (5+3) * (4+3) = 56
    assert ctx.estimate_complexity("y ~ te(s(a, k=5, deg=3), s(b, k=4, deg=3))") == 56


def test_estimate_complexity_additive_sums_terms():
    ctx = _ctx()
    # l(x)=1 + s(y, k=10, deg=3)=13 -> 14
    assert ctx.estimate_complexity("y ~ l(x) + s(y, k=10, deg=3)") == 14


def test_estimate_complexity_invalid_formula_returns_one():
    ctx = _ctx()
    assert ctx.estimate_complexity("not a formula") == 1
    assert ctx.estimate_complexity("") == 1


def test_estimate_complexity_defaults_when_params_absent():
    ctx = _ctx()
    # l(x) with no k param -> default k=1
    k = ctx.estimate_complexity("y ~ l(x)")
    assert k >= 1


# --------------------------------------------------------------------------- #
# penalize_score
# --------------------------------------------------------------------------- #

def test_penalize_score_increases_with_complexity():
    ctx = _ctx(penalty=1.0)
    base = ctx.penalize_score(5.0, "y ~ l(x)", n_samples=100)       # k=1
    heavy = ctx.penalize_score(5.0, "y ~ s(x, k=10, deg=3)", n_samples=100)  # k=13
    assert heavy > base


def test_penalize_score_with_zero_samples_returns_inf():
    ctx = _ctx()
    assert ctx.penalize_score(5.0, "y ~ l(x)", n_samples=0) == float("inf")


def test_penalize_score_inf_score_propagates():
    ctx = _ctx()
    assert ctx.penalize_score(float("inf"), "y ~ l(x)", n_samples=100) == float("inf")
