# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Strong regularisation (opt-in ``strong_regularization``): the strong penalties are searched, the margins of a tensor product take strong penalties,
and a champion that fits its folds almost perfectly is also fitted with more knots and the adaptive penalty ``m_pen=4``.
"""

import types

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.pipeline.expert_expander import ExpertExpander
from tam.model.autotam.population_nodes import STRONG_MARGIN_AP

EXPANSIONS = {"prior": True, "autofit": True, "kalman": False, "adaptive": False, "grid": False}


def _curve(n=150, noise=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": np.sin(np.pi * x) + noise * rng.normal(size=n)})


def _space(strong, n=150):
    frame = _curve(n)
    return EffectSelector(strong_regularization=strong).build_search_space(frame, {"targets": ["y"], "features": ["x", "z"]}, {})


def test_the_default_grid_has_one_penalty_and_the_strong_grid_has_the_strong_ones():
    default, strong = _space(False), _space(True)
    assert len(default["x"]["grids"]["s"]["ap"]) == 1 and "strong_regularization" not in default
    assert {-5.0, -3.0, -1.0, 0.0, -6.0} <= set(strong["x"]["grids"]["s"]["ap"]) and strong["strong_regularization"] is True
    assert -6.0 not in _space(True, n=60)["x"]["grids"]["s"]["ap"]                   # the weak penalty only from 100 rows


def test_the_margins_of_a_tensor_product_take_a_strong_penalty():
    import random
    from tam.model.autotam.knowledge_graph import KnowledgeGraph
    from tam.model.autotam.population_nodes import _tensor_term
    space = _space(True)
    for feature in ("x", "z"):
        space[feature]["eligible_effects"] = ["l", "s", "p"]
    random.seed(0)
    term = _tensor_term(KnowledgeGraph(), space, "x", "z", family="s")
    assert term is not None and any(f"ap={ap}" in term for ap in STRONG_MARGIN_AP)


def _context(frame):
    ctx = types.SimpleNamespace(target="y", group_col=None, date_col=None, cv_folds=[(frame.iloc[:100], frame.iloc[100:])],
                                penalize_score=lambda score, formula, n: score, estimate_complexity=lambda f: 1)
    return ctx


def test_a_clean_curve_gets_more_knots_and_the_adaptive_penalty():
    frame = _curve(n=150, noise=0.01)
    expander = ExpertExpander(EXPANSIONS)
    ctx = _context(frame)
    pool = {"Static_X_R1_Prior": {"cv_rmse": 0.02, "type": "static", "island": "SplineIsland"}}
    expander._high_snr_experts(pool, [], ctx, "X_R1", "SplineIsland", "y ~ s(x, k=6, ap=-3.0)", frame, len(frame), "rmse")
    assert "Static_X_R1_HighSNR_Prior" in pool and "Static_X_R1_HighSNR_Auto" in pool
    formula = pool["Static_X_R1_HighSNR_Prior"]["model"].formula_.replace(" ", "")
    assert "k=30" in formula and "m_pen=4" in formula


def test_a_noisy_curve_or_a_short_sample_gets_nothing_more():
    frame = _curve(n=150, noise=1.0)
    expander = ExpertExpander(EXPANSIONS)
    pool = {"Static_X_R1_Prior": {"cv_rmse": 0.9, "type": "static", "island": "SplineIsland"}}
    expander._high_snr_experts(pool, [], _context(frame), "X_R1", "SplineIsland", "y ~ s(x, k=6)", frame, len(frame), "rmse")
    assert len(pool) == 1
    clean = _curve(n=40, noise=0.01)
    pool = {"Static_X_R1_Prior": {"cv_rmse": 0.02, "type": "static", "island": "SplineIsland"}}
    expander._high_snr_experts(pool, [], _context(clean), "X_R1", "SplineIsland", "y ~ s(x, k=6)", clean, len(clean), "rmse")
    assert len(pool) == 1


def test_the_option_reaches_the_selector_and_the_context(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x, z)", n_experts=1, pop_size=4, use_opera=False, seed=1, strong_regularization=True, export_dir=str(tmp_path))
    assert model.data_manager.selector.strong_regularization is True and model.strong_regularization is True
    assert AutoTAM("y ~ AutoPipe(x, z)", export_dir=str(tmp_path / "d")).data_manager.selector.strong_regularization is False
