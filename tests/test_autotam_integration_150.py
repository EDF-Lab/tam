# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
What TAM 1.5.0 added, used by AutoTAM: the complexity of the new effect parameters, the debiased native quantile expert, REML for the auto_fit experts,
target-transform experts, adaptive experts on a family base, and the effect parameters searched with ``extended_effects``.
"""

import types

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.native_quantiles import DISTRIBUTION, NativeQuantileExpert
from tam.model.autotam.pipeline.context import PipelineContext
from tam.model.autotam.pipeline.expert_expander import ExpertExpander
from tam.model.additive import StaticTAM

EXPANSIONS = {"prior": True, "autofit": True, "kalman": False, "adaptive": True, "grid": False}


def test_the_complexity_of_the_thin_plate_spline_and_the_graph_follows_their_columns():
    ctx = PipelineContext
    estimate = lambda f: ctx.estimate_complexity(types.SimpleNamespace(), f)
    assert estimate("y ~ rbf(x, others='z', n_centers=30)") == 30 and estimate("y ~ rbf(x, others='z', n_centers=30, kernel='tps')") == 29
    assert estimate("y ~ c(r, n_cat=10, topo='nominal')") == 9 and estimate("y ~ c(r, n_cat=10, topo='graph', coords='a:b')") == 10
    assert estimate("y ~ n(x, n_neurons=10, orthogonalize=True)") == estimate("y ~ n(x, n_neurons=10)")


def test_the_native_quantile_expert_uses_the_debiased_scale_and_the_weighted_location():
    assert DISTRIBUTION["scale_correction"] == "leverage" and DISTRIBUTION["refit_rounds"] == 2
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0, 1, n)
    frame = pd.DataFrame({"x": x, "y": 5 + 2 * x + rng.normal(size=n) * (0.2 + 1.5 * x)})
    expert = NativeQuantileExpert("y", "y ~ s(x, k=6)").fit(frame)
    q = expert.predict_quantiles(frame, [0.1, 0.9])
    inside = float(((frame["y"] >= q["q0.1"]) & (frame["y"] <= q["q0.9"])).mean())
    assert 0.74 <= inside <= 0.86, inside
    width = (q["q0.9"] - q["q0.1"]).to_numpy()
    assert width[x > 0.8].mean() > 1.8 * width[x < 0.2].mean()                       # the intervals widen where the noise does


def _ctx(frame, criterion="gcv"):
    return types.SimpleNamespace(target="y", group_col=None, date_col=None, autofit_criterion=criterion, cv_folds=[(frame.iloc[:100], frame.iloc[100:])],
                                 penalize_score=lambda score, formula, n: score, estimate_complexity=lambda f: 1,
                                 target_profile={"hints": ["log"]})


def test_the_auto_fit_experts_pass_reml_only_when_asked():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 150)})
    frame["y"] = np.sin(3 * frame["x"]) + rng.normal(scale=0.2, size=150)
    assert ExpertExpander._criterion(_ctx(frame, "gcv")) == {} and ExpertExpander._criterion(_ctx(frame, "reml")) == {"criterion": "reml"}
    StaticTAM("y ~ s(x, k=8)").auto_fit(frame, **ExpertExpander._criterion(_ctx(frame, "reml")))


def test_a_positive_skewed_target_gets_a_log_transform_expert_back_on_its_own_scale():
    rng = np.random.default_rng(2)
    n = 150
    x = rng.uniform(0, 1, n)
    frame = pd.DataFrame({"x": x, "y": np.exp(1 + 1.2 * x + rng.normal(scale=0.3, size=n))})
    pool = {}
    ExpertExpander(EXPANSIONS)._transform_experts(pool, [], _ctx(frame), "X_R1", "SplineIsland", "y ~ s(x, k=6)", frame, n, "rmse")
    assert "Static_X_R1_Transform_log" in pool and np.isfinite(pool["Static_X_R1_Transform_log"]["cv_rmse"])
    prediction = pool["Static_X_R1_Transform_log"]["model"].predict(frame)["Estimatedy"].to_numpy()
    assert prediction.min() > 0 and 1.0 < prediction.mean() < 10.0                      # the scale of the target, not of its logarithm
    skipped = {}
    ExpertExpander(EXPANSIONS)._transform_experts(skipped, [], _ctx(frame.assign(y=frame["y"] - 20)), "X_R1", "I", "y ~ s(x, k=6)", frame.assign(y=frame["y"] - 20), n, "rmse")
    assert skipped == {}                                                                  # a log needs a positive target


def test_the_best_family_expert_is_the_one_that_beats_the_gaussian_expert():
    pool = {"Static_P_R1_Family_poisson": {"model": "m1", "cv_rmse": 0.8}, "Static_P_R1_FamilyAuto_poisson": {"model": "m2", "cv_rmse": 0.7},
            "Static_P_R1_Prior": {"model": "m0", "cv_rmse": 1.0}, "Static_Q_R1_Family_gamma": {"model": "m3", "cv_rmse": 0.1}}
    assert ExpertExpander._best_family(pool, "P_R1", 1.0) == ("m2", "poisson", 0.7)
    assert ExpertExpander._best_family(pool, "P_R1", 0.6) is None


def test_the_extended_grids_offer_the_new_parameters_only_when_asked():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 200), "z": rng.integers(0, 4, 200).astype(float), "y": rng.normal(size=200)})
    config = {"targets": ["y"], "features": ["x", "z"]}
    plain, extended = EffectSelector().build_search_space(frame, config, {}), EffectSelector(extended_effects=True).build_search_space(frame, config, {})
    assert "kernel" not in plain["x"]["grids"]["rbf"] and "m_pen" not in plain["x"]["grids"]["s"] and "extended_effects" not in plain
    assert extended["x"]["grids"]["rbf"]["kernel"] == ["matern", "tps"] and extended["x"]["grids"]["s"]["m_pen"] == [1, 2, 4]
    assert extended["x"]["grids"]["n"]["orthogonalize"] == [False, True] and extended["x"]["grids"]["t"]["orthogonalize"] == [False, True]
    assert extended["z"]["grids"]["c"]["unseen"] == ["edge", "mean", "zero"] and "ordinal" in extended["z"]["grids"]["c"]["topo"]
    assert extended["extended_effects"] is True


def test_the_arguments_reach_the_context_and_a_bad_criterion_is_refused(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x)", autofit_criterion="reml", transform_experts=True, family_adaptive=True, extended_effects=True, export_dir=str(tmp_path))
    assert model.autofit_criterion == "reml" and model.transform_experts and model.family_adaptive and model.data_manager.selector.extended_effects
    with pytest.raises(ValueError, match="autofit_criterion"):
        AutoTAM("y ~ AutoPipe(x)", autofit_criterion="aic", export_dir=str(tmp_path / "b"))
