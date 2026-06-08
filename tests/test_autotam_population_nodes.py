r"""
Unit tests for ``tam.model.autotam.population_nodes``, the evolutionary islands
that generate candidate formula strings by querying the Knowledge Graph.
"""

import random

import pytest

from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.population_nodes import (
    BaseIsland, LinearIsland, SplineIsland, FourierIsland, ChebyshevIsland,
    WaveletIsland, NeuralIsland, RBFIsland, TreeIsland, CrossIsland,
    SmallContinent, Continent,
    get_island_generators, _clean_and_join_terms, _get_safe_params,
)


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
    assert len(generators) == 11
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
    LinearIsland, SplineIsland, NeuralIsland, CrossIsland, SmallContinent, Continent,
])
def test_islands_return_intercept_without_features(island_cls):
    island = island_cls()
    assert island.generate(KnowledgeGraph(), [], _search_space()) == "1"


def test_cross_island_generates_tensor_or_simple_terms():
    random.seed(3)
    island = CrossIsland()
    rhs = island.generate(KnowledgeGraph(), ["temp", "humidity"], _search_space())
    assert isinstance(rhs, str) and rhs != ""


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
