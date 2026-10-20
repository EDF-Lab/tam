# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Crossover (opt-in): a mutation may take one branch from the champion of another island, and may insert a tensor product of a used feature
with another feature. Both respect the covariate lock and the eligible effects of the search space.
"""

import random

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.population_nodes import Continent, LinearIsland

GRIDS = {"s": {"k": [10], "deg": [3], "p": [2], "ap": [-5.0]}, "f": {"m": [6], "s": [2], "ap": [-5.0]}, "l": {"ap": [-9.0]},
         "c": {"n_cat": [5], "topo": ["nominal"], "p_order": [1]}}


def _space(features, smooth=()):
    space = {}
    for f in features:
        space[f] = {"eligible_effects": ["l", "s", "f"], "grids": GRIDS, "max_active_effects": 2, "topology": "continuous",
                    "profile": {"linear": f not in smooth}}
    return space


def _island(crossover, donors, champion=None, cls=Continent):
    island = cls()
    island.crossover = crossover
    island.set_champion(champion or [{"type": "l", "feature": "x", "params": {}}])
    island.donor_genomes = donors
    return island


def _draws(island, available, space, n=250, seed=1):
    random.seed(seed)
    kg = KnowledgeGraph()
    return [island.mutate(kg, available, space) for _ in range(n)]


def test_a_donor_branch_is_taken_only_with_the_option():
    donors = [[{"type": "s", "feature": "z", "params": {"k": 17, "deg": 3, "p": 2, "ap": -5.0}}]]   # k = 17 is not in the grid
    space = _space(["x", "z"])
    off = _draws(_island(False, donors), ["x", "z"], space)
    on = _draws(_island(True, donors), ["x", "z"], space)
    assert not any("k=17" in f for f in off)
    assert any("k=17" in f for f in on)


def test_a_donor_branch_on_a_feature_that_is_not_available_is_never_taken():
    donors = [[{"type": "l", "feature": "secret", "params": {}}]]
    space = _space(["x", "z", "secret"])
    formulas = _draws(_island(True, donors), ["x", "z"], space)
    assert not any("secret" in f for f in formulas)


def test_a_donor_effect_that_is_not_eligible_for_its_feature_is_never_taken():
    donors = [[{"type": "w", "feature": "z", "params": {"n_scales": 2, "n_locations": 8, "ap": 0.0}}]]
    space = _space(["x", "z"])           # w is not eligible
    assert not any("w(z" in f for f in _draws(_island(True, donors), ["x", "z"], space))


def test_a_full_genome_loses_a_term_to_take_the_branch():
    champion = [{"type": "l", "feature": f, "params": {}} for f in ("x", "y", "z")]   # every feature used: no plain insertion is possible
    donors = [[{"type": "s", "feature": "z", "params": {"k": 17, "deg": 3, "p": 2, "ap": -5.0}}]]
    island = _island(True, donors, champion=champion)
    island.max_terms = 3
    space = _space(["x", "y", "z"])
    formulas = _draws(island, ["x", "y", "z"], space)
    taken = [f for f in formulas if "k=17" in f]
    assert taken and all(len([t for t in f.split(" + ") if t.strip()]) <= 3 for f in taken)


def test_a_tensor_product_of_a_used_non_linear_feature_is_inserted_only_with_the_option():
    space = _space(["x", "z"], smooth=("x", "z"))
    off = _draws(_island(False, [], cls=LinearIsland), ["x", "z"], space)
    on = _draws(_island(True, [], cls=Continent), ["x", "z"], space)
    assert not any("te(" in f for f in off)
    assert any("te(" in f for f in on)


def test_the_default_mutation_is_unchanged_by_the_new_attributes():
    space = _space(["x", "z"])
    donors = [[{"type": "s", "feature": "z", "params": {"k": 10, "deg": 3, "p": 2, "ap": -5.0}}]]
    a = _draws(_island(False, donors), ["x", "z"], space, seed=7)
    b = _draws(_island(False, []), ["x", "z"], space, seed=7)
    assert a == b


def test_the_argument_reaches_the_islands_and_a_fit_runs(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1, x2)", n_experts=2, pop_size=6, use_opera=False, seed=3, crossover=True, export_dir=str(tmp_path))
    assert model.discoverer.crossover is True
    assert AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path)).discoverer.crossover is False
    rng = np.random.default_rng(4)
    n = 160
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": rng.normal(size=n), "x2": rng.uniform(size=n)})
    frame["y"] = 4 + frame["x1"] + np.sin(6 * frame["x2"]) + rng.normal(scale=0.2, size=n)
    model.fit(frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert model.trained_experts and model.architectures_

