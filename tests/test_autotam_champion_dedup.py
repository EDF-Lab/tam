# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Champions that differ only by blanks or by the order of their terms are one model: the island keeps the best of them and the next distinct formula
takes the other's place; and an expert that repeats another (same formula, same expansion) is not fitted and weighted twice.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.base_discoverer import BaseDiscoverer
from tam.model.autotam.pipeline.expert_expander import ExpertExpander


def test_formulas_that_differ_by_blanks_are_one_champion():
    evaluated = [("SplineIsland", "y ~ te(p(Time, 3), s(temp, 5) )", 1.0), ("SplineIsland", "y ~ te(p(Time, 3), s(temp, 5))", 1.0),
                 ("SplineIsland", "y ~ l(Time)", 2.0)]
    assert BaseDiscoverer.top_formulas(evaluated) == {"SplineIsland": ["y ~ te(p(Time, 3), s(temp, 5) )", "y ~ l(Time)"]}


def test_the_order_of_the_terms_does_not_make_another_champion_and_the_best_ranked_is_kept():
    evaluated = [("LinearIsland", "y ~ l(b) + l(a)", 3.0), ("LinearIsland", "y ~ l(a) + l(b)", 2.0), ("LinearIsland", "y ~ l(c)", 5.0)]
    assert BaseDiscoverer.top_formulas(evaluated) == {"LinearIsland": ["y ~ l(a) + l(b)", "y ~ l(c)"]}


def test_the_next_distinct_formula_fills_the_slot_and_each_island_keeps_its_own():
    evaluated = [("A", f"y ~ l(x{i % 3})", float(i)) for i in range(9)] + [("B", "y ~ l(z)", 1.0)]
    out = BaseDiscoverer.top_formulas(evaluated, per_island=5)
    assert out["A"] == ["y ~ l(x0)", "y ~ l(x1)", "y ~ l(x2)"] and out["B"] == ["y ~ l(z)"]


def _expert(name, island, formula="y ~ l(x1)", kind="static"):
    return {"type": kind, "island": island, "model": SimpleNamespace(formula_=formula), "params": {}}


def test_an_expert_that_repeats_another_is_left_out():
    pool = {"Static_LinearIsland_R1_Prior": _expert("a", "LinearIsland"), "Static_SplineIsland_R1_Prior": _expert("b", "SplineIsland"),
            "Static_LinearIsland_R2_Auto": _expert("c", "LinearIsland"), "Static_SplineIsland_R2_Prior": _expert("d", "SplineIsland", "y ~ l(x2)")}
    assert list(ExpertExpander._without_duplicates(pool)) == ["Static_LinearIsland_R1_Prior", "Static_LinearIsland_R2_Auto", "Static_SplineIsland_R2_Prior"]


def test_a_guard_island_expert_is_never_left_out_and_the_search_expert_gives_way_to_it():
    pool = {"Static_LinearIsland_R1_Prior": _expert("a", "LinearIsland"), "Static_Seed_R1_Prior": _expert("c", "Seed")}
    assert list(ExpertExpander._without_duplicates(pool)) == ["Static_Seed_R1_Prior"]


def test_a_fit_has_no_numbered_identifier(tmp_path):
    rng = np.random.default_rng(5)
    n = 200
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    frame["y"] = 5 + 2 * frame["x1"] + rng.normal(scale=0.3, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=3, pop_size=8, use_opera=True, seed=2, export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False},
              seed_formulas=["y ~ l(x1)"])
    assert (model.experts_["identifier"].str.len() == 13).all() and model.experts_["identifier"].is_unique      # no _2, _3 suffix
