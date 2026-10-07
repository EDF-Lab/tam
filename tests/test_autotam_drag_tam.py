# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

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
        if available_features:
            return f"l({available_features[0]})"
        return self._next()

    def mutate(self, kg, available_features, search_space):
        if available_features:
            return f"s({available_features[0]})"
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


def test_dragtam_ensure_mandatory_constraints_injects_missing_terms():
    engine = DragTAM(target_col="y", mandatory_terms=["s(temp, k=10)"])
    counter = itertools.count()
    island = _StubIsland("A", counter)
    res = engine._ensure_mandatory_constraints("l(x1)", island)
    # Must contain both terms
    assert "l(x1)" in res
    assert "s(temp, k=10)" in res


def test_dragtam_ensure_mandatory_constraints_does_not_duplicate_existing_terms():
    engine = DragTAM(target_col="y", mandatory_terms=["s(temp, k=10)"])
    counter = itertools.count()
    island = _StubIsland("A", counter)
    res = engine._ensure_mandatory_constraints("l(x1) + s(temp, k=10)", island)
    assert res.count("s(temp, k=10)") == 1


def test_dragtam_all_evaluated_candidates_contain_mandatory_terms():
    random.seed(0)
    engine = DragTAM(target_col="y", population_size=6, n_generations=3, mandatory_terms=["s(temp, k=10)"])
    evaluated = []

    def fake_evaluate(formula, cv_folds, metric="rmse"):
        evaluated.append(formula)
        return 1.0, None, parse_formula_to_terms(formula)[1], {}

    engine._evaluate_candidate = fake_evaluate
    counter = itertools.count()
    folds = [(pd.DataFrame({"y": np.arange(10.0)}), pd.DataFrame({"y": np.arange(10.0)}))]
    best_formula = engine.optimize(
        cv_folds=folds,
        islands=[_StubIsland("A", counter), _StubIsland("B", counter)],
        search_space={}
    )

    assert len(evaluated) > 0
    for formula in evaluated:
        assert "s(temp, k=10)" in formula, f"Formula missing mandatory term: {formula}"
    assert "s(temp, k=10)" in best_formula

def test_dragtam_ensure_mandatory_constraints_injects_missing_variables():
    engine = DragTAM(target_col="y", mandatory_variables=["temp"])
    counter = itertools.count()
    island = _StubIsland("A", counter)
    res = engine._ensure_mandatory_constraints("l(x1)", island)
    assert "l(x1)" in res
    assert "l(temp)" in res

def test_dragtam_ensure_mandatory_constraints_does_not_duplicate_existing_variables():
    engine = DragTAM(target_col="y", mandatory_variables=["temp"])
    counter = itertools.count()
    island = _StubIsland("A", counter)
    res = engine._ensure_mandatory_constraints("l(x1) + l(temp)", island)
    assert res.count("l(temp)") == 1

def test_dragtam_all_evaluated_candidates_contain_mandatory_variables():
    random.seed(0)
    engine = DragTAM(target_col="y", population_size=6, n_generations=3, mandatory_variables=["temp"])
    evaluated = []

    def fake_evaluate(formula, cv_folds, metric="rmse"):
        evaluated.append(formula)
        try:
            terms = parse_formula_to_terms(formula)[1]
        except ValueError:
            terms = []
        return 1.0, None, terms, {}

    engine._evaluate_candidate = fake_evaluate
    counter = itertools.count()
    folds = [(pd.DataFrame({"y": np.arange(10.0)}), pd.DataFrame({"y": np.arange(10.0)}))]
    best_formula = engine.optimize(
        cv_folds=folds,
        islands=[_StubIsland("A", counter), _StubIsland("B", counter)],
        search_space={'a0': {}, 'b0': {}, 'a1': {}, 'b1': {}, 'a2': {}, 'b2': {}, 'temp': {}}
    )

    assert len(evaluated) > 0
    for formula in evaluated:
        assert "temp" in formula, f"Formula missing mandatory variable: {formula}"
    assert "temp" in best_formula

def test_dragtam_all_evaluated_candidates_contain_mandatory_terms_and_variables():
    random.seed(0)
    engine = DragTAM(
        target_col="y",
        population_size=6,
        n_generations=3,
        mandatory_terms=["s(x, k=5)"],
        mandatory_variables=["z"]
    )
    evaluated = []

    def fake_evaluate(formula, cv_folds, metric="rmse"):
        evaluated.append(formula)
        try:
            terms = parse_formula_to_terms(formula)[1]
        except ValueError:
            terms = []
        return 1.0, None, terms, {}

    engine._evaluate_candidate = fake_evaluate
    counter = itertools.count()
    folds = [(pd.DataFrame({"y": np.arange(10.0)}), pd.DataFrame({"y": np.arange(10.0)}))]
    best_formula = engine.optimize(
        cv_folds=folds,
        islands=[_StubIsland("A", counter), _StubIsland("B", counter)],
        search_space={'a0': {}, 'b0': {}, 'a1': {}, 'b1': {}, 'a2': {}, 'b2': {}, 'x': {}, 'z': {}}
    )

    assert len(evaluated) > 0
    for formula in evaluated:
        assert "s(x, k=5)" in formula, f"Formula missing mandatory term: {formula}"
        assert "z" in formula, f"Formula missing mandatory variable: {formula}"
    assert "s(x, k=5)" in best_formula
    assert "z" in best_formula


def test_dragtam_ensure_mandatory_constraints_invariant_matching_and_verbatim_retention():
    # Candidate term differs only in parameter ordering
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["c(WeekDays, topo='nominal', n_cat=7)"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # Candidate has reversed parameter order
    candidate_rhs = "c(WeekDays, n_cat=7, topo='nominal') + l(x1)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    # Must retain verbatim representation of mandatory term, not duplicate it
    assert "c(WeekDays, topo='nominal', n_cat=7)" in res
    assert "c(WeekDays, n_cat=7, topo='nominal')" not in res
    assert "l(x1)" in res


def test_dragtam_ensure_mandatory_constraints_tensor_invariant_matching_and_verbatim_retention():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["te(s(x1), s(x2))"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # Candidate has reversed sub-terms
    candidate_rhs = "te(s(x2), s(x1)) + l(z)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    assert "te(s(x1), s(x2))" in res
    assert "te(s(x2), s(x1))" not in res
    assert "l(z)" in res


class _MultiTermIsland:
    """Island that generates multiple terms joined by ' + ' for a feature."""
    def __init__(self, name):
        self.name = name
        self.champion_genome = None
        self.last_search_space = None

    def generate(self, kg, available_features, search_space, complexity_cap=False):
        self.last_search_space = search_space
        feat = available_features[0] if available_features else "x"
        return f"s({feat}, k=10) + l({feat})"

    def mutate(self, kg, available_features, search_space):
        return self.generate(kg, available_features, search_space)

    def set_champion(self, genome):
        self.champion_genome = genome


def test_dragtam_ensure_mandatory_constraints_multi_term_island_generation():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["s(temp, k=10)"],
        mandatory_variables=["humidity"]
    )
    island = _MultiTermIsland("Multi")
    res = engine._ensure_mandatory_constraints("l(x1)", island, search_space={"humidity": {}})

    assert "s(temp, k=10)" in res
    assert "l(x1)" in res
    assert "s(humidity, k=10)" in res
    assert "l(humidity)" in res
    assert island.last_search_space == {"humidity": {}}
    # Ensure formula has balanced parentheses
    assert res.count("(") == res.count(")")
    # Must be validly split by _split_rhs_terms
    from tam.model.autotam.drag_tam import _split_rhs_terms
    parts = _split_rhs_terms(res)
    assert len(parts) == 4



def test_dragtam_ensure_mandatory_constraints_evicts_non_mandatory_lifo_on_effects_cap():
    # Feature 'x' has 2 candidate terms, then mandatory term is injected bringing count to 3.
    # Non-mandatory terms should be evicted in LIFO order (rightmost added dropped first).
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["f(x, m=3)"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # Leftmost is s(x, k=5), rightmost is l(x)
    candidate_rhs = "s(x, k=5) + l(x) + l(other)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    # f(x, m=3) must be present (mandatory, protected)
    assert "f(x, m=3)" in res
    # s(x, k=5) is leftmost, so it should survive
    assert "s(x, k=5)" in res
    # l(x) was rightmost, so it should be evicted
    assert "l(x)" not in res
    # l(other) is for another feature, unaffected
    assert "l(other)" in res


def test_dragtam_ensure_mandatory_constraints_protects_mandatory_terms_when_evicting():
    # If a feature has 2 mandatory terms and candidate has a non-mandatory term,
    # the non-mandatory term must be evicted and both mandatory terms preserved.
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["s(x, k=10)", "l(x)"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    candidate_rhs = "c(x, n_cat=3) + l(other)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    assert "s(x, k=10)" in res
    assert "l(x)" in res
    assert "c(x, n_cat=3)" not in res
    assert "l(other)" in res


def test_dragtam_build_canonical_formula_deduplication_and_verbatim_retention():
    engine = DragTAM(
        target_col="Load",
        mandatory_terms=["c(WeekDays, topo='nominal', n_cat=7)"]
    )

    # RHS contains equivalent duplicate terms differing in parameter order
    rhs = "c(WeekDays, n_cat=7, topo='nominal') + c(WeekDays, topo='nominal', n_cat=7) + s(Temp, k=10)"
    formula = engine._build_canonical_formula(rhs)

    # Must preserve verbatim mandatory representation and deduplicate
    assert formula == "Load ~ c(WeekDays, topo='nominal', n_cat=7) + s(Temp, k=10)"


def test_dragtam_ablation_loop_respects_canonical_mandatory_terms():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["c(WeekDays, topo='nominal', n_cat=7)"]
    )
    # Check that canonical mandatory terms property is exposed and populated
    assert hasattr(engine, "_canonical_mandatory_terms")
    assert "c(WeekDays, n_cat=7, topo='nominal')" in engine._canonical_mandatory_terms

    # Simulate survivor whose pruned_terms has rearranged parameters
    survivor_terms = [
        {"type": "c", "feature": "WeekDays", "params": {"n_cat": 7, "topo": "nominal"}},
        {"type": "l", "feature": "x1", "params": {}}
    ]
    # Variance order: mandatory term has lower variance (would be ablated first without protection)
    engine.kg.term_avg_variance = lambda t: 0.1 if t["feature"] == "WeekDays" else 1.0

    sorted_terms = sorted(survivor_terms, key=engine.kg.term_avg_variance)
    from tam.model.autotam.parser import canonicalize_term
    ablated_terms = None
    for i in range(len(sorted_terms)):
        term_to_ablate = sorted_terms[i]
        if canonicalize_term(term_to_ablate) not in engine._canonical_mandatory_terms:
            ablated_terms = sorted_terms[:i] + sorted_terms[i+1:]
            break

    assert ablated_terms is not None
    assert len(ablated_terms) == 1
    assert ablated_terms[0]["feature"] == "WeekDays"


def test_dragtam_search_space_isolation_excludes_external_variables():
    from tam.model.autotam.effect_selector import EffectSelector
    from tam.model.autotam.pipeline.data_manager import DataManager

    df = pd.DataFrame({
        "y": np.random.randn(50),
        "temp": np.random.randn(50),
        "humidity": np.random.randn(50),
        "pressure": np.random.randn(50),
    })

    dm = DataManager(
        formula="y ~ s(humidity, k=10) + AutoPipe(temp)",
    )
    ctx = dm.prepare(df_train=df)

    # Search space must be built exclusively from AutoPipe features
    selector = EffectSelector()
    search_space = selector.build_search_space(ctx.df_all_aug, ctx.formula_config, ctx.metadata)

    assert "temp" in search_space
    # External variable 'humidity' must be strictly isolated from search_space
    assert "humidity" not in search_space
    assert "pressure" not in search_space


def test_dragtam_ensure_mandatory_constraints_subsumption_matching_preserves_tuned_free_params():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["s(temp, k=10)"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # Candidate has tuned free parameter m=2 on top of mandatory fixed parameter k=10
    candidate_rhs = "s(temp, k=10, m=2) + l(x1)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    # Must preserve candidate's tuned free parameter m=2 and NOT inject s(temp, k=10) as duplicate
    assert "s(temp, k=10, m=2)" in res
    assert res.count("temp") == 1
    assert "s(temp, k=10)" not in res or "s(temp, k=10, m=2)" in res
    assert "l(x1)" in res


def test_dragtam_ensure_mandatory_constraints_tensor_subsumption_and_order_invariance():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["te(s(x1, k=5), s(x2))"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # Candidate has reversed sub-terms and tuned free parameter on s(x2)
    candidate_rhs = "te(s(x2, k=10), s(x1, k=5)) + l(z)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    # Subsumption match must succeed, preserving tuned free params and preventing duplicate injection
    assert res.count("te(") == 1
    assert "te(s(x2, k=10), s(x1, k=5))" in res or "te(s(x1, k=5), s(x2, k=10))" in res
    assert "l(z)" in res


def test_dragtam_ensure_mandatory_constraints_eviction_protects_subsumed_mandatory_terms():
    # A candidate has 2 non-mandatory terms for feature 'x' and 1 subsumed mandatory term.
    # Total active effects for 'x' = 3 (exceeds cap of 2).
    # The subsumed mandatory term must be protected, and the rightmost non-mandatory evicted.
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["s(x, k=10)"]
    )
    counter = itertools.count()
    island = _StubIsland("A", counter)

    # s(x, k=10, m=2) subsumes mandatory s(x, k=10).
    # l(x) is first non-mandatory (left), c(x, n_cat=3) is second non-mandatory (right).
    candidate_rhs = "l(x) + s(x, k=10, m=2) + c(x, n_cat=3) + l(other)"
    res = engine._ensure_mandatory_constraints(candidate_rhs, island)

    # s(x, k=10, m=2) must be preserved because it subsumes mandatory specification
    assert "s(x, k=10, m=2)" in res
    # l(x) survives as leftmost non-mandatory
    assert "l(x)" in res
    # c(x, n_cat=3) was rightmost non-mandatory, evicted by LIFO
    assert "c(x, n_cat=3)" not in res
    assert "l(other)" in res


def test_dragtam_ablation_loop_skips_subsumed_mandatory_terms():
    engine = DragTAM(
        target_col="y",
        mandatory_terms=["s(x, k=10)"]
    )
    # Candidate term s(x, k=10, m=2) subsumes s(x, k=10)
    survivor_terms = [
        {"type": "s", "feature": "x", "params": {"k": 10, "m": 2}},
        {"type": "l", "feature": "z", "params": {}}
    ]
    # Make the subsumed mandatory term have lower variance so it would be ablated first without protection
    engine.kg.term_avg_variance = lambda t: 0.05 if t["feature"] == "x" else 1.0

    from tam.model.autotam.parser import term_subsumes
    sorted_terms = sorted(survivor_terms, key=engine.kg.term_avg_variance)
    ablated_terms = None
    for i in range(len(sorted_terms)):
        term_to_ablate = sorted_terms[i]
        if not any(term_subsumes(term_to_ablate, m) for m in engine.mandatory_terms):
            ablated_terms = sorted_terms[:i] + sorted_terms[i+1:]
            break

    assert ablated_terms is not None
    assert len(ablated_terms) == 1
    # l(z) was ablated; s(x, k=10, m=2) was protected
    assert ablated_terms[0]["feature"] == "x"


def test_knowledge_graph_update_and_prune_exempts_subsumed_mandatory_terms():
    from tam.model.autotam.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(prune_threshold=0.10)

    # Candidate term has tuned free parameter m=2
    parsed_terms = [
        {"type": "s", "feature": "temp", "params": {"k": 10, "m": 2}},
        {"type": "l", "feature": "noise", "params": {}},
    ]

    class FakeModel:
        def decompose_prediction(self, df):
            # temp has 0 variance (below threshold), noise has 0 variance
            return {
                "effect_temp": np.zeros(len(df)),
                "effect_noise": np.zeros(len(df)),
            }
        def predict(self, df):
            return pd.DataFrame({"Estimatedy": np.zeros(len(df))})

    df_val = pd.DataFrame({"y": np.zeros(10), "temp": np.zeros(10), "noise": np.zeros(10)})

    pruned = kg.update_and_prune(
        parsed_terms=parsed_terms,
        model=FakeModel(),
        df=df_val,
        target_col="y",
        global_rmse=1.0,
        target_std=1.0,
        mandatory_terms=["s(temp, k=10)"]
    )

    features_in_pruned = [t["feature"] for t in pruned]
    # temp must be kept because s(temp, k=10, m=2) subsumes mandatory s(temp, k=10)
    assert "temp" in features_in_pruned
    # noise should be pruned due to negligible variance
    assert "noise" not in features_in_pruned


