# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Tensor products open to every Island (``population_nodes._choose_effect`` and ``_tensor_term``).

A product crosses a continuous feature diagnosed non-linear either with a categorical of at least
two levels, at the feature's profiled capacity, or with another such continuous feature, through a
small fixed basis on each side (at most 25 columns).
"""

import random
import re

import pytest

from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.pipeline.context import term_complexity
from tam.model.autotam.population_nodes import (
    CONTINUOUS_TENSOR_BASES, MAX_TENSOR_TERMS, SMOOTH_INTERACTION_BASES,
    FourierIsland, SplineIsland, get_island_objects,
)

TE = re.compile(r"^te\((\w+)\((\w+)(.*?)\),\s*(\w+)\((\w+)(.*)\)\)$")


def _continuous(m, k):
    return {"topology": "continuous", "profile": {"linear": False, "m": m, "k": k},
            "eligible_effects": ["l", "s", "p", "w", "n", "rbf", "t", "f", "lt"],
            "grids": {"l": {"ap": [-9.0]}, "f": {"m": [m], "s": [2], "ap": [-3.0]},
                      "s": {"k": [k], "deg": [3], "p": [2], "ap": [-5.0]},
                      "p": {"deg": [12], "s": [2], "ap": [-5.0]},
                      "w": {"n_scales": [4], "n_locations": [13], "ap": [-5.0]}}}


def _space():
    """Search space shaped like EffectSelector's output on national load."""
    return {
        "toy": _continuous(22, 40),
        "temperature": _continuous(8, 16),
        "Load_d1": {"topology": "continuous", "profile": {"linear": True}, "eligible_effects": ["l", "pid"],
                    "grids": {"l": {"ap": [-30.0]}, "pid": {"w": [3, 7], "d_pen": [10.0]}}},
        "day_type_week": {"topology": "discrete", "profile": None, "eligible_effects": ["l", "c", "t"],
                          "grids": {"l": {"ap": [-9.0]}, "c": {"n_cat": [7], "topo": ["nominal"], "p_order": [1]}}},
        "crise_covid": {"topology": "discrete", "profile": None, "eligible_effects": ["l", "c", "t"],
                        "grids": {"l": {"ap": [-9.0]}, "c": {"n_cat": [1], "topo": ["nominal"], "p_order": [1]}}},
    }


def _formulas(island_cls, space=None, draws=300, seed=0):
    random.seed(seed)
    space = space or _space()
    return [island_cls().generate(KnowledgeGraph(exploration_rate=1.0), list(space), space) for _ in range(draws)]


def _tensors(rhs):
    return [term for term in rhs.split(" + ") if term.startswith("te(")]


def _sides(term):
    match = TE.match(term)
    assert match, term
    return (match.group(1), match.group(2), match.group(3)), (match.group(4), match.group(5), match.group(6))


def _width(side):
    effect, feature, params = side
    return term_complexity(parse_formula_to_terms(f"y ~ {effect}({feature}{params})")[1][0])


@pytest.mark.parametrize("island_cls", [type(island) for island in get_island_objects()])
def test_every_island_builds_tensor_products(island_cls):
    assert any(_tensors(rhs) for rhs in _formulas(island_cls))


def test_cross_island_is_gone():
    assert "CrossIsland" not in {island.name for island in get_island_objects()}


@pytest.mark.parametrize("island_cls", [SplineIsland, FourierIsland])
def test_a_categorical_product_keeps_the_profiled_capacity_and_the_island_basis(island_cls):
    family = island_cls.tensor_family
    categorical = [_sides(t) for rhs in _formulas(island_cls) for t in _tensors(rhs) if ", c(" in t]
    assert categorical
    for (effect, feature, params), (partner_effect, partner, _) in categorical:
        assert feature in {"toy", "temperature"} and partner == "day_type_week" and partner_effect == "c"
        assert effect == family
        capacity = {"s": {"toy": "k=40", "temperature": "k=16"}, "f": {"toy": "m=22", "temperature": "m=8"}}
        assert capacity[family][feature] in params


def test_two_continuous_sides_take_the_small_fixed_basis():
    surfaces = [_sides(t) for island in get_island_objects() for rhs in _formulas(type(island), draws=150)
                for t in _tensors(rhs) if ", c(" not in t]
    assert surfaces
    for left, right in surfaces:
        assert {left[1], right[1]} == {"toy", "temperature"}
        assert left[0] in CONTINUOUS_TENSOR_BASES and right[0] in CONTINUOUS_TENSOR_BASES
        assert _width(left) * _width(right) <= 25


def test_products_never_start_on_linear_locked_or_discrete_features_and_all_parse():
    for island in get_island_objects():
        for rhs in _formulas(type(island), draws=100):
            parse_formula_to_terms(f"y ~ {rhs}")
            for term in _tensors(rhs):
                (_, feature, _), (_, partner, _) = _sides(term)
                assert feature in {"toy", "temperature"}
                assert partner not in {"Load_d1", "crise_covid"}


def test_at_most_max_tensor_terms_per_formula():
    for island in get_island_objects():
        assert all(len(_tensors(rhs)) <= MAX_TENSOR_TERMS for rhs in _formulas(type(island), draws=100))


def test_no_partner_means_no_product():
    space = _space()
    for name in ("temperature", "day_type_week"):
        space.pop(name)
    for island in get_island_objects():
        assert not any(_tensors(rhs) for rhs in _formulas(type(island), space=space, draws=100))


def test_smooth_bases_for_categorical_products_include_chebyshev():
    assert set(SMOOTH_INTERACTION_BASES) == {"f", "s", "w", "p"}
