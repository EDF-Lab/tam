# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Island selection of the Mutant-UCB bandit (``KnowledgeGraph.select_island_ucb``) and its use by
DragTAM: minimum budget, epsilon-greedy draws, failed fits counted as zero-reward pulls, and
non-viable Islands left out.

Candidate evaluation is stubbed, so the loop runs in milliseconds.
"""

import itertools
import random

import numpy as np
import pandas as pd

from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.knowledge_graph import KnowledgeGraph


def _kg_with_pulls(pulls):
    kg = KnowledgeGraph()
    for name, rewards in pulls.items():
        for reward in rewards:
            kg.update_island_reward(name, reward)
    return kg


def test_islands_below_the_minimum_budget_are_selected_first():
    kg = _kg_with_pulls({"A": [5.0, 5.0, 5.0], "B": [1.0]})
    random.seed(0)
    assert {kg.select_island_ucb(["A", "B"], min_pulls=3) for _ in range(20)} == {"B"}


def test_epsilon_draws_reach_islands_ucb_would_never_pick():
    kg = _kg_with_pulls({"A": [6.0] * 10, "B": [1.0] * 10})
    random.seed(0)
    assert {kg.select_island_ucb(["A", "B"]) for _ in range(200)} == {"A"}
    draws = [kg.select_island_ucb(["A", "B"], epsilon=0.5) for _ in range(400)]
    assert 0.15 < draws.count("B") / len(draws) < 0.35       # expected share 0.25


def test_defaults_keep_plain_ucb_with_unqueried_islands_first():
    kg = _kg_with_pulls({"A": [6.0]})
    random.seed(0)
    assert kg.select_island_ucb(["A", "B"]) == "B"


def test_a_failed_fit_counts_as_a_zero_reward_pull():
    kg = _kg_with_pulls({"A": [4.0, float("inf"), None]})
    assert kg.island_stats["A"]["N"] == 3
    assert kg.island_stats["A"]["reward_sum"] == 4.0


class _StubIsland:
    """Island yielding a never-repeating stream of one-term formulas named after the Island."""

    def __init__(self, name, counter, viable=True):
        self.name = name
        self.champion_genome = None
        self._counter = counter
        self._viable = viable

    def is_viable(self, available_features, search_space):
        return self._viable

    def generate(self, kg, available_features, search_space, complexity_cap=False):
        return f"l({self.name}_x{next(self._counter)})"

    def mutate(self, kg, available_features, search_space):
        return self.generate(kg, available_features, search_space)

    def set_champion(self, genome):
        self.champion_genome = genome


def _optimize(islands, n_generations=8):
    random.seed(0)
    engine = DragTAM(target_col="y", population_size=4, n_generations=n_generations)
    engine._evaluate_candidate = lambda formula, folds, metric="rmse": (
        float("inf") if "Broken" in formula else 1.0, None, parse_formula_to_terms(formula)[1], {})
    frame = pd.DataFrame({"y": np.arange(10.0)})
    engine.optimize(cv_folds=[(frame, frame)], islands=islands, search_space={})
    return engine


def test_dragtam_spends_the_minimum_budget_and_counts_failed_fits():
    counter = itertools.count()
    engine = _optimize([_StubIsland(name, counter) for name in ("Good", "Other", "Broken")])
    stats = engine.kg.island_stats
    assert all(stats[name]["N"] >= engine.ucb_min_pulls for name in ("Good", "Other", "Broken"))
    # Once its budget is spent, an Island whose fits all fail holds zero reward and is rarely drawn.
    assert stats["Broken"]["reward_sum"] == 0.0
    assert stats["Broken"]["N"] < stats["Good"]["N"] + stats["Other"]["N"]


def test_dragtam_leaves_non_viable_islands_out_of_the_bandit():
    counter = itertools.count()
    engine = _optimize([_StubIsland("Viable", counter), _StubIsland("Dead", counter, viable=False)],
                       n_generations=3)
    assert engine.kg.island_stats["Viable"]["N"] > 0
    assert "Dead" not in engine.kg.island_stats
