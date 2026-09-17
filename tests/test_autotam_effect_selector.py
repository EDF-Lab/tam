# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.effect_selector.EffectSelector``, builds the
mathematically safe search space and enforces the Strict Covariate Lock.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.feature_profiler import FeatureProfiler


def test_topology_non_numeric_is_discrete():
    sel = EffectSelector()
    assert sel._analyze_topology(pd.Series(["a", "b", "c", "a"])) == "discrete"


def test_topology_low_cardinality_is_discrete():
    sel = EffectSelector(categorical_threshold=15)
    assert sel._analyze_topology(pd.Series([0, 1, 2, 0, 1, 2])) == "discrete"


def test_topology_sparse_detection():
    sel = EffectSelector(sparsity_threshold=0.80)
    values = [0.0] * 80 + list(np.linspace(1, 20, 20))  # 80% zeros, >15 unique
    assert sel._analyze_topology(pd.Series(values)) == "sparse"


def test_topology_continuous_detection():
    sel = EffectSelector()
    assert sel._analyze_topology(pd.Series(np.linspace(0, 100, 200))) == "continuous"


def test_rolling_feature_name_forces_continuous():
    sel = EffectSelector()
    series = pd.Series([0, 0, 0, 1], name="temp_rolling_mean_3_steps")
    assert sel._analyze_topology(series) == "continuous"


def test_build_search_space_continuous_feature():
    # A continuous feature with a curved response unlocks the rich spectral/non-linear effect set.
    # The feature is not monotone in row order: the FeatureProfiler treats a strictly increasing
    # column as an index or trend and locks it to a linear basis.
    rng = np.random.default_rng(0)
    temp = rng.uniform(0, 50, 200)
    df = pd.DataFrame({"load": 100 + 30 * np.sin(temp / 4.0) + rng.normal(0, 1, 200), "temp": temp})
    sel = EffectSelector()
    space = sel.build_search_space(
        df, config={"targets": ["load"], "features": ["temp"]},
        metadata={"date_col": None, "group_col": None},
    )
    assert "temp" in space
    assert space["temp"]["topology"] == "continuous"
    for eff in ["s", "f", "p", "w", "n", "rbf", "t"]:
        assert eff in space["temp"]["eligible_effects"]


def test_build_search_space_locks_a_linear_response_to_l():
    # The FeatureProfiler hard-locks a feature whose response is linear: non-linear bases would
    # only fit noise while taking search budget.
    rng = np.random.default_rng(0)
    temp = rng.uniform(0, 50, 200)
    df = pd.DataFrame({"load": 100 + 2.0 * temp + rng.normal(0, 1, 200), "temp": temp})
    sel = EffectSelector()
    space = sel.build_search_space(
        df, config={"targets": ["load"], "features": ["temp"]},
        metadata={"date_col": None, "group_col": None},
    )
    assert space["temp"]["eligible_effects"] == ["l"]


def test_build_search_space_discrete_feature_unlocks_categorical():
    df = pd.DataFrame({"load": np.linspace(0, 100, 60), "weekday": ([0, 1, 2, 3, 4, 5] * 10)})
    sel = EffectSelector()
    space = sel.build_search_space(
        df, config={"targets": ["load"], "features": ["weekday"]},
        metadata={"date_col": None, "group_col": None},
    )
    assert space["weekday"]["topology"] == "discrete"
    assert "c" in space["weekday"]["eligible_effects"]


def test_covariate_lock_accepts_within_limit():
    sel = EffectSelector(max_active_effects=2)
    assert sel.validate_covariate_lock(["s(temp)", "l(temp)"]) is True


def test_covariate_lock_rejects_over_limit():
    sel = EffectSelector(max_active_effects=2)
    assert sel.validate_covariate_lock(["s(temp)", "l(temp)", "f(temp)"]) is False


def _space_for(df, feature):
    return EffectSelector().build_search_space(
        df, config={"targets": ["load"], "features": [feature]},
        metadata={"date_col": None, "group_col": None},
    )


def test_autoregressive_lag_may_take_a_pid_term():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 301))
    space = _space_for(pd.DataFrame({"load": series[1:], "load_lag_1": series[:-1]}), "load_lag_1")
    assert space["load_lag_1"]["eligible_effects"] == ["l", "pid"]


def test_trend_stays_locked_to_l():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"load": rng.normal(0, 1, 300), "trend": np.arange(300.0)})
    assert _space_for(df, "trend")["trend"]["eligible_effects"] == ["l"]


def test_continuous_grids_offer_tree_architectures_linear_trees_and_activations():
    rng = np.random.default_rng(0)
    temp = rng.uniform(0, 50, 200)
    df = pd.DataFrame({"load": 100 + 30 * np.sin(temp / 4.0) + rng.normal(0, 1, 200), "temp": temp})
    space = _space_for(df, "temp")["temp"]
    assert "lt" in space["eligible_effects"]
    assert set(space["grids"]["n"]["act"]) == {"relu", "tanh"}
    assert {"max_depth", "max_leaves", "split_strategy", "sp_alpha"} <= set(space["grids"]["t"])


def test_only_the_most_recent_lag_may_take_a_pid_term():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 308))
    df = pd.DataFrame({"load": series[7:], "load_lag_1": series[6:-1], "load_lag_7": series[:-7]})
    space = EffectSelector().build_search_space(
        df, config={"targets": ["load"], "features": ["load_lag_1", "load_lag_7"]},
        metadata={"date_col": None, "group_col": None},
    )
    assert space["load_lag_1"]["eligible_effects"] == ["l", "pid"]
    assert space["load_lag_7"]["eligible_effects"] == ["l"]


def test_most_recent_lag_ignores_derived_lag_features():
    features = ["Load_d7", "Load_d1_ewma_alpha30", "trend", "Load_d1"]
    assert FeatureProfiler.lag_order("Load_d1_ewma_alpha30") is None
    assert FeatureProfiler.most_recent_lag(features) == "Load_d1"
    assert FeatureProfiler.most_recent_lag(["trend", "temperature"]) is None
