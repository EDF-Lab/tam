r"""
Unit tests for ``tam.model.autotam.effect_selector.EffectSelector``, builds the
mathematically safe search space and enforces the Strict Covariate Lock.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.effect_selector import EffectSelector


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
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"load": rng.normal(100, 10, 200), "temp": np.linspace(0, 50, 200)})
    sel = EffectSelector()
    space = sel.build_search_space(
        df, config={"targets": ["load"], "features": ["temp"]},
        metadata={"date_col": None, "group_col": None},
    )
    assert "temp" in space
    assert space["temp"]["topology"] == "continuous"
    # Continuous features unlock the rich spectral/non-linear effect set.
    for eff in ["s", "f", "p", "w", "n", "rbf", "t"]:
        assert eff in space["temp"]["eligible_effects"]


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
