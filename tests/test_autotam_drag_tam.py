# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.drag_tam.DragTAM``: evidence accounting and the repopulation
budget of the evolutionary loop.

Candidate evaluation is stubbed (no StaticTAM fits), so the loop runs in milliseconds; the fits
themselves are covered elsewhere.
"""

import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd

from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.drag_tam import DragTAM


class _StubIsland:
    """Island yielding a never-repeating stream of two-term formulas."""

    def __init__(self, name, counter):
        self.name = name
        self.champion_genome = None
        self._counter = counter

    def _next(self):
        i = next(self._counter)
        return f"l(a{i}) + l(b{i})"

    def generate(self, kg, available_features, search_space, complexity_cap=False):
        return self._next()

    def mutate(self, kg, available_features, search_space):
        return self._next()

    def set_champion(self, genome):
        self.champion_genome = genome


def _run(population_size=8, n_generations=6):
    """Runs the loop with generation-0 formulas scoring best forever, so the same elites persist."""
    random.seed(0)
    engine = DragTAM(target_col="y", population_size=population_size, n_generations=n_generations)
    evaluated = []

    def fake_evaluate(formula, cv_folds, metric="rmse"):
        generation = len(engine.history)
        evaluated.append((generation, formula))
        return (1.0 if generation == 0 else 2.0), None, parse_formula_to_terms(formula)[1], {}

    prune_calls = Counter()
    original_prune = engine.kg.update_and_prune

    def counting_prune(*args, **kwargs):
        prune_calls["n"] += 1
        return original_prune(*args, **kwargs)

    engine._evaluate_candidate = fake_evaluate
    engine.kg.update_and_prune = counting_prune

    counter = itertools.count()
    folds = [(pd.DataFrame({"y": np.arange(10.0)}), pd.DataFrame({"y": np.arange(10.0)}))]
    engine.optimize(cv_folds=folds, islands=[_StubIsland("A", counter), _StubIsland("B", counter)],
                    search_space={})
    return engine, evaluated, prune_calls["n"]


def test_each_distinct_formula_is_evaluated_and_credited_once():
    engine, evaluated, prune_calls = _run()
    formulas = [formula for _, formula in evaluated]

    assert len(formulas) == len(set(formulas))
    # Cached elites carried across generations must not be re-pruned, re-rewarded, or re-pulled.
    assert prune_calls == len(formulas)
    assert sum(stats["N"] for stats in engine.kg.island_stats.values()) == len(formulas)


def test_every_generation_evaluates_fresh_candidates_while_elites_persist():
    population_size, n_generations = 8, 6
    engine, evaluated, _ = _run(population_size, n_generations)

    new_per_generation = Counter(generation for generation, _ in evaluated)
    n_fresh = max(1, round(population_size * engine.fresh_fraction))
    # Without the reserved share, elites plus their (cached) ablations filled the population and
    # these generations evaluated nothing new.
    for generation in range(1, n_generations):
        assert new_per_generation[generation] >= n_fresh, dict(new_per_generation)


def test_survival_is_credited_once_per_formula():
    engine, _, _ = _run()
    # Four identical gen-0 elites survive all six generations; each is credited a single time.
    assert max(node["survival_count"] for node in engine.kg.features.values()) == 1
