# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The extended search space (opt-in): wider grids for spline knots, RBF centres and the PID window, and constant features left out. The default
space is the one it always was.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_selector import EffectSelector


def _frame(n=1200):
    rng = np.random.default_rng(0)
    temp = rng.uniform(0, 50, n)
    return pd.DataFrame({"load": 100 + 30 * np.sin(temp / 4.0) + rng.normal(0, 1, n), "temp": temp, "flat": 3.0})


def _space(extended):
    return EffectSelector(extended=extended).build_search_space(
        _frame(), config={"targets": ["load"], "features": ["temp", "flat"]}, metadata={"date_col": None, "group_col": None})


def test_the_default_space_is_unchanged():
    space = _space(False)
    grids = space["temp"]["grids"]
    assert len(grids["s"]["k"]) == 1 and len(grids["rbf"]["n_centers"]) == 1 and grids["pid"]["w"] == [3, 7]
    assert "flat" in space


def test_the_extended_space_offers_more_knots_and_centres_up_to_fifty_and_a_wider_pid_grid():
    base, wide = _space(False)["temp"]["grids"], _space(True)["temp"]["grids"]
    assert len(wide["s"]["k"]) > 1 and max(wide["s"]["k"]) <= 50 and set(base["s"]["k"]) <= set(wide["s"]["k"])
    assert len(wide["rbf"]["n_centers"]) > 1 and max(wide["rbf"]["n_centers"]) <= 50 and set(base["rbf"]["n_centers"]) <= set(wide["rbf"]["n_centers"])
    assert wide["pid"]["w"] == [3, 6, 7]


def test_a_constant_feature_is_left_out_only_in_the_extended_space():
    assert "flat" in _space(False) and "flat" not in _space(True) and "temp" in _space(True)


def test_the_knots_stay_within_the_rows_of_a_short_series():
    short = _frame(120)
    space = EffectSelector(extended=True).build_search_space(short, config={"targets": ["load"], "features": ["temp"]},
                                                              metadata={"date_col": None, "group_col": None})
    assert max(space["temp"]["grids"]["s"]["k"]) <= max(10, 120 // 20) or max(space["temp"]["grids"]["s"]["k"]) <= 10


def test_the_argument_reaches_the_effect_selector(tmp_path):
    assert AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path)).data_manager.selector.extended is False
    assert AutoTAM("y ~ AutoPipe(x1)", extended_search=True, export_dir=str(tmp_path)).data_manager.selector.extended is True


# --------------------------------------------------------------------------- #
# Second-basis mutation
# --------------------------------------------------------------------------- #

def _mutations(extended, draws=300, max_active=2):
    import random

    from tam.model.autotam.knowledge_graph import KnowledgeGraph
    from tam.model.autotam.population_nodes import Continent

    random.seed(1)
    island = Continent()
    island.extended_mutations = extended
    island.set_champion([{"type": "l", "feature": "x", "params": {}}, {"type": "l", "feature": "z", "params": {}}])
    grids = {"s": {"k": [10], "deg": [3], "p": [2], "ap": [-5.0]}, "f": {"m": [6], "s": [2], "ap": [-5.0]}, "l": {"ap": [-9.0]}}
    space = {"x": {"eligible_effects": ["l", "s", "f"], "grids": grids, "max_active_effects": max_active},
             "z": {"eligible_effects": ["l"], "grids": grids, "max_active_effects": max_active}}
    kg = KnowledgeGraph()
    return [island.mutate(kg, ["x", "z"], space) for _ in range(draws)]


def test_a_second_basis_on_a_used_feature_appears_only_in_the_extended_search():
    def has_second(formula):
        return "l(x" in formula and ("s(x" in formula or "f(x" in formula)
    assert not any(has_second(f) for f in _mutations(False))
    assert any(has_second(f) for f in _mutations(True))


def test_the_second_basis_respects_the_covariate_lock():
    assert not any(("s(x" in f or "f(x" in f) and "l(x" in f for f in _mutations(True, max_active=1))
