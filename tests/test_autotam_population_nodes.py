# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.population_nodes``, the evolutionary islands
that generate candidate formula strings by querying the Knowledge Graph.
"""

import random
import re

import pytest

from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.population_nodes import (
    BaseIsland, LinearIsland, SplineIsland, FourierIsland, ChebyshevIsland,
    WaveletIsland, NeuralIsland, RBFIsland, TreeIsland,
    SmallContinent, Continent,
    get_island_generators, _clean_and_join_terms, _get_safe_params, MAX_INTERACTION_PARTNERS,
)
from tam.model.autotam import population_nodes


def _search_space():
    return {
        "temp": {
            "eligible_effects": ["l", "s", "f", "n", "rbf", "t", "te"],
            "topology": "continuous",
            "grids": {"s": {"k": [5, 10, 20]}, "f": {"m": [3, 5]}, "n": {"n_neurons": [8, 16]}},
        },
        "humidity": {
            "eligible_effects": ["l", "s", "n", "rbf", "t", "te"],
            "topology": "continuous",
            "grids": {"s": {"k": [5, 10]}},
        },
    }


# ------------------------------- helper functions -------------------------- #

def test_clean_and_join_deduplicates_terms():
    assert _clean_and_join_terms(["l(x)", "l(x)", "s(y)"]) == "l(x) + s(y)"


def test_clean_and_join_empty_returns_intercept():
    assert _clean_and_join_terms([]) == "1"
    assert _clean_and_join_terms(["1", ""]) == "1"


def test_get_safe_params_complexity_cap_selects_minimum():
    kg = KnowledgeGraph()
    params = _get_safe_params(kg, _search_space(), "temp", "s", complexity_cap=True)
    assert params["k"] == 5  # minimum of [5, 10, 20]


def test_get_safe_params_falls_back_to_random_grid_sample():
    kg = KnowledgeGraph()  # empty -> suggest_parameters returns {}
    random.seed(0)
    params = _get_safe_params(kg, _search_space(), "temp", "s", complexity_cap=False)
    assert params["k"] in [5, 10, 20]


# ------------------------------- base contract ----------------------------- #

def test_base_island_generate_is_abstract():
    island = BaseIsland("base", ["l"])
    with pytest.raises(NotImplementedError):
        island.generate(KnowledgeGraph(), ["temp"], _search_space())


def test_island_registry_returns_all_generators():
    generators = get_island_generators()
    assert len(generators) == 10
    assert all(callable(g) for g in generators)


# ------------------------------- generation -------------------------------- #

@pytest.mark.parametrize("island_cls", [
    LinearIsland, SplineIsland, FourierIsland, ChebyshevIsland,
    WaveletIsland, NeuralIsland, RBFIsland, TreeIsland,
])
def test_standard_and_interaction_islands_generate_valid_rhs(island_cls):
    random.seed(0)
    island = island_cls()
    rhs = island.generate(KnowledgeGraph(), ["temp", "humidity"], _search_space())
    assert isinstance(rhs, str) and len(rhs) > 0


@pytest.mark.parametrize("island_cls", [
    LinearIsland, SplineIsland, NeuralIsland, SmallContinent, Continent,
])
def test_islands_return_intercept_without_features(island_cls):
    island = island_cls()
    assert island.generate(KnowledgeGraph(), [], _search_space()) == "1"


def test_interaction_island_injects_others_param():
    """Forcing the interaction effect wires an 'others' covariate into the term."""
    kg = KnowledgeGraph(exploration_rate=0.0)
    space = {
        "temp": {"eligible_effects": ["n"], "topology": "continuous", "grids": {"n": {}}},
        "humidity": {"eligible_effects": ["n"], "topology": "continuous", "grids": {"n": {}}},
    }
    random.seed(0)
    rhs = NeuralIsland().generate(kg, ["temp", "humidity"], space)
    assert "n(" in rhs  # at least one neural term was emitted


def test_small_continent_and_continent_compose_terms():
    random.seed(5)
    space = _search_space()
    small = SmallContinent().generate(KnowledgeGraph(), ["temp", "humidity"], space)
    big = Continent().generate(KnowledgeGraph(), ["temp", "humidity"], space)
    assert isinstance(small, str) and isinstance(big, str)


# ------------------------------- deep-island partners ---------------------- #

def _partner_space():
    """A non-linear continuous base with six two-level categoricals and one constant flag."""
    space = {"temp": {"eligible_effects": ["l", "n", "rbf", "t"], "topology": "continuous",
                      "profile": {"linear": False}, "grids": {"n": {"n_neurons": [16]}}}}
    for name in ["a", "b", "c", "d", "e", "f"]:
        space[name] = {"eligible_effects": ["l", "c", "t"], "topology": "discrete",
                       "grids": {"c": {"n_cat": [2]}}}
    space["flag"] = {"eligible_effects": ["l", "c", "t"], "topology": "discrete",
                     "grids": {"c": {"n_cat": [1]}}}
    return space


def _deep_terms_on_temp(island, draws=400):
    space = _partner_space()
    terms = []
    for _ in range(draws):
        rhs = island.generate(KnowledgeGraph(exploration_rate=1.0), list(space), space)
        terms += [t for t in rhs.split(" + ") if t.startswith(("n(temp", "rbf(temp", "t(temp"))]
    return terms


def _partners(term):
    match = re.search(r"others='?([A-Za-z0-9_|]+)", term)
    return match.group(1).split("|") if match else []


@pytest.mark.parametrize("island_cls", [NeuralIsland, RBFIsland, TreeIsland])
def test_deep_island_partners_are_never_constant_categoricals(island_cls):
    random.seed(0)
    partnered = [t for t in _deep_terms_on_temp(island_cls()) if _partners(t)]
    assert partnered
    assert all("flag" not in _partners(t) for t in partnered)


def test_deep_island_draws_several_distinct_sorted_partners():
    random.seed(1)
    counts = []
    for term in _deep_terms_on_temp(NeuralIsland()):
        partners = _partners(term)
        if partners:
            assert partners == sorted(set(partners))
            counts.append(len(partners))
    assert 1 < max(counts) <= MAX_INTERACTION_PARTNERS


def test_deep_island_plain_term_share_follows_the_probability(monkeypatch):
    monkeypatch.setattr(population_nodes, "PLAIN_DEEP_TERM_PROBABILITY", 1.0)
    random.seed(2)
    plain = _deep_terms_on_temp(TreeIsland(), draws=100)
    assert plain and not any(_partners(t) for t in plain)

    monkeypatch.setattr(population_nodes, "PLAIN_DEEP_TERM_PROBABILITY", 0.0)
    random.seed(2)
    partnered = _deep_terms_on_temp(TreeIsland(), draws=100)
    assert partnered and all(_partners(t) for t in partnered)



def test_tree_island_proposes_linear_trees_and_linear_island_proposes_pid():
    space = _partner_space()
    space["temp"]["eligible_effects"].append("lt")
    space["Load_d1"] = {"eligible_effects": ["l", "pid"], "topology": "continuous",
                        "profile": {"linear": True}, "grids": {"pid": {"w": [3, 7], "d_pen": [10.0]}}}
    random.seed(4)
    tree = [TreeIsland().generate(KnowledgeGraph(exploration_rate=1.0), list(space), space) for _ in range(300)]
    assert any("lt(temp" in rhs for rhs in tree)
    linear = [LinearIsland().generate(KnowledgeGraph(exploration_rate=1.0), ["Load_d1"], space) for _ in range(50)]
    assert any(rhs.startswith("pid(Load_d1") for rhs in linear)


def test_get_safe_params_keeps_one_tree_architecture():
    kg = KnowledgeGraph()
    space = {"x": {"grids": {"t": {"n_trees": [10], "max_depth": [1, 2], "max_leaves": [50]}}}}
    random.seed(0)
    for _ in range(50):
        params = _get_safe_params(kg, space, "x", "t")
        assert ("max_depth" in params) != ("max_leaves" in params)
    assert _get_safe_params(kg, space, "x", "t", complexity_cap=True) == {"n_trees": 10, "max_depth": 1}

# ------------------------------- mutation ---------------------------------- #

def test_mutate_protects_mandatory_term_from_deletion():
    random.seed(0)
    island = LinearIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["l"]},
        "y": {"eligible_effects": ["l"]},
        "mandatory_terms": ["l(x)"]
    }
    
    champion_genome = [
        {'type': 'l', 'feature': 'x', 'params': {}},
        {'type': 'l', 'feature': 'y', 'params': {}}
    ]
    island.set_champion(champion_genome)

    for _ in range(100):
        mutated_formula = island.mutate(kg, ["x", "y"], search_space)
        assert "l(x)" in mutated_formula

def test_mutate_protects_mandatory_term_from_modification():
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["s"], "grids": {"s": {"k": [5, 10]}}},
        "y": {"eligible_effects": ["l"]},
        "mandatory_terms": ["s(x, k=5)"]
    }
    
    champion_genome = [
        {'type': 's', 'feature': 'x', 'params': {'k': 5}},
        {'type': 'l', 'feature': 'y', 'params': {}}
    ]
    island.set_champion(champion_genome)

    for _ in range(100):
        mutated_formula = island.mutate(kg, ["x", "y"], search_space)
        assert "s(x, k=10)" not in mutated_formula

def test_mutate_protects_sole_mandatory_variable_term_from_deletion():
    random.seed(0)
    island = LinearIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["l"]},
        "mandatory_variables": ["x"]
    }
    
    champion_genome = [{'type': 'l', 'feature': 'x', 'params': {}}]
    island.set_champion(champion_genome)
    
    for _ in range(50):
        mutated_formula = island.mutate(kg, ["x"], search_space)
        assert "l(x)" in mutated_formula

def test_mutate_allows_deleting_non_sole_mandatory_variable_term():
    random.seed(0)
    island = LinearIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["l", "s"], "grids": {"s": {"k": [5]}}},
        "mandatory_variables": ["x"]
    }
    
    champion_genome = [
        {'type': 'l', 'feature': 'x', 'params': {}},
        {'type': 's', 'feature': 'x', 'params': {'k': 5}}
    ]
    island.set_champion(champion_genome)

    deleted = False
    for _ in range(100):
        mutated_formula = island.mutate(kg, ["x"], search_space)
        if "l(x)" not in mutated_formula or "s(x, k=5)" not in mutated_formula:
            deleted = True
            break
    assert deleted, "A non-sole mandatory variable term should be deletable"


# ----------------------- Ticket 004: Canonical Deduplication & Mutation ----------------------- #

def test_clean_and_join_deduplicates_terms_canonically():
    res = _clean_and_join_terms([
        "c(WeekDays, topo='nominal', n_cat=7)",
        "c(WeekDays, n_cat=7, topo='nominal')",
    ])
    assert res == "c(WeekDays, topo='nominal', n_cat=7)"

    res_te = _clean_and_join_terms([
        "te(s(x, k=5), c(y, n_cat=2))",
        "te(c(y, n_cat=2), s(x, k=5))",
    ])
    assert res_te == "te(s(x, k=5), c(y, n_cat=2))"


def test_clean_and_join_substitutes_verbatim_mandatory_term():
    res = _clean_and_join_terms(
        ["c(WeekDays, n_cat=7, topo='nominal')"],
        mandatory_terms=["c(WeekDays, topo='nominal', n_cat=7)"],
    )
    assert res == "c(WeekDays, topo='nominal', n_cat=7)"

    res_dedup = _clean_and_join_terms(
        ["c(WeekDays, n_cat=7, topo='nominal')", "c(WeekDays, topo='nominal', n_cat=7)"],
        mandatory_terms=["c(WeekDays, topo='nominal', n_cat=7)"],
    )
    assert res_dedup == "c(WeekDays, topo='nominal', n_cat=7)"


def test_clean_and_join_enforces_covariate_lock_and_tensor_cap():
    res_feat = _clean_and_join_terms(["s(x, k=5)", "l(x)", "f(x, m=3)"])
    assert res_feat == "s(x, k=5) + l(x)"

    res_te = _clean_and_join_terms(["te(s(x), c(y))", "te(s(z), c(w))", "te(s(a), c(b))"])
    assert res_te == "te(s(x), c(y)) + te(s(z), c(w))"


def test_mutate_protects_mandatory_term_under_parameter_permutation():
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["s"], "grids": {"s": {"k": [5, 10], "deg": [2, 3]}}},
        "y": {"eligible_effects": ["l"]},
        "mandatory_terms": ["s(x, deg=2, k=5)"],
    }

    # Champion genome has k=5, deg=2 (which in _genome_to_rhs sorted params would be deg=2, k=5)
    champion_genome = [
        {'type': 's', 'feature': 'x', 'params': {'k': 5, 'deg': 2}},
        {'type': 'l', 'feature': 'y', 'params': {}},
    ]
    island.set_champion(champion_genome)

    for _ in range(100):
        mutated_formula = island.mutate(kg, ["x", "y"], search_space)
        # Term s(x, ...) must be present, must never be mutated to k=10 or deg=3, and must keep verbatim representation
        assert "s(x, deg=2, k=5)" in mutated_formula
        assert "s(x, deg=3" not in mutated_formula
        assert "k=10" not in mutated_formula


# ------------------- Ticket 003: Parameter-Constrained Mutation ------------------- #

def test_mutate_protects_subsumed_mandatory_term_with_free_params_from_deletion():
    """A mandatory term with tuned free parameters must never be deleted."""
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["s"], "grids": {"s": {"k": [5]}}},
        "y": {"eligible_effects": ["l"], "grids": {"l": {}}},
        "mandatory_terms": ["s(x, k=5)"],
    }

    # Champion genome has s(x) with tuned free parameter deg=3 alongside k=5, plus a deletable l(y)
    champion_genome = [
        {'type': 's', 'feature': 'x', 'params': {'k': 5, 'deg': 3}},
        {'type': 'l', 'feature': 'y', 'params': {}},
    ]
    island.set_champion(champion_genome)

    for _ in range(100):
        mutated_formula = island.mutate(kg, ["x", "y"], search_space)
        # s(x, ...) subsumes s(x, k=5) and must NEVER be deleted
        assert "s(x" in mutated_formula
        assert "k=5" in mutated_formula


def test_mutate_locks_user_fixed_params_while_tuning_free_params():
    """Mutation must never alter fixed params, but may tune unconstrained free keys."""
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {
            "eligible_effects": ["s"],
            "grids": {"s": {"k": [5, 10, 20], "deg": [2, 3, 4]}},
        },
        "mandatory_terms": ["s(x, k=5)"],
    }

    champion_genome = [
        {'type': 's', 'feature': 'x', 'params': {'k': 5, 'deg': 2}},
    ]
    island.set_champion(champion_genome)

    deg_tuned = False
    for _ in range(100):
        mutated = island.mutate(kg, ["x"], search_space)
        # Fixed parameter k=5 must NEVER be mutated to 10 or 20
        assert "k=10" not in mutated
        assert "k=20" not in mutated
        assert "k=5" in mutated
        # Free parameter deg can be tuned
        if "deg=3" in mutated or "deg=4" in mutated:
            deg_tuned = True

    assert deg_tuned, "Free parameter 'deg' should have been tuned across mutations"


def test_mutate_external_variable_absent_from_search_space_uses_fallback_grid():
    """External variables absent from search_space must use fallback effect grids to tune free params."""
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    # humidity is NOT in search_space (it's an external mandatory variable)
    search_space = {
        "temp": {"eligible_effects": ["l"], "grids": {"l": {}}},
        "mandatory_terms": ["s(humidity, k=10)"],
    }

    champion_genome = [
        {'type': 'l', 'feature': 'temp', 'params': {}},
        {'type': 's', 'feature': 'humidity', 'params': {'k': 10}},
    ]
    island.set_champion(champion_genome)

    free_param_tuned = False
    for _ in range(100):
        mutated = island.mutate(kg, ["temp"], search_space)
        # humidity must never be deleted
        assert "s(humidity" in mutated
        # User-fixed k=10 must never be altered
        assert "k=10" in mutated
        # Free parameters from fallback grid (e.g. deg or p or ap) can be tuned
        if "deg=" in mutated or "p=" in mutated or "ap=" in mutated:
            free_param_tuned = True

    assert free_param_tuned, "External mandatory variable should tune free parameters via fallback grid"


def test_mutate_fully_immutable_when_all_grid_params_fixed():
    """When all grid parameters are fixed by the mandatory specification, term cannot be modified."""
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["s"], "grids": {"s": {"k": [5, 10], "deg": [2, 3]}}},
        "mandatory_terms": ["s(x, deg=2, k=5)"],
    }

    champion_genome = [
        {'type': 's', 'feature': 'x', 'params': {'k': 5, 'deg': 2}},
    ]
    island.set_champion(champion_genome)

    for _ in range(50):
        mutated = island.mutate(kg, ["x"], search_space)
        # With no unused features, single term, and no free keys, formula remains unchanged
        assert mutated == "s(x, deg=2, k=5)"


def test_mutate_mandatory_tensor_term_protected_from_deletion():
    """Mandatory tensor interaction terms must be protected from deletion in mutate()."""
    random.seed(0)
    island = SplineIsland()
    kg = KnowledgeGraph()
    search_space = {
        "x": {"eligible_effects": ["s"], "grids": {"s": {"k": [5]}}},
        "y": {"eligible_effects": ["c"], "grids": {"c": {"n_cat": [2]}}},
        "z": {"eligible_effects": ["l"], "grids": {"l": {}}},
        "mandatory_terms": ["te(s(x, k=5), c(y, n_cat=2))"],
    }

    champion_genome = [
        {
            'type': 'te',
            'feature': 'interaction',
            'params': {'s(x, deg=2, k=5)': 1, 'c(y, n_cat=2)': 1},
        },
        {'type': 'l', 'feature': 'z', 'params': {}},
    ]
    island.set_champion(champion_genome)

    for _ in range(100):
        mutated = island.mutate(kg, ["x", "y", "z"], search_space)
        # Mandatory tensor product must never be deleted
        assert "te(" in mutated
        assert "s(x" in mutated
        assert "c(y" in mutated


