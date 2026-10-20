# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
General additive modelling with AutoTAM (opt-in ``smooth_monotonic``, ``cross_sectional``): a sorted predictor is not a time counter, a time counter may
take a low-degree smooth, and data with no time order are split at random and see no Kalman or adaptive expert.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.feature_profiler import FeatureProfiler

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _sorted_curve(n=300, seed=0):
    """A response that bends, in a table sorted by the predictor (mcycle-like)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 6, n))
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": np.sin(x) * 3 + rng.normal(scale=0.2, size=n)})


def test_the_default_forces_a_sorted_predictor_to_a_line():
    frame = _sorted_curve()
    profile = FeatureProfiler().profile(frame, "y", ["x", "z"])
    assert profile["x"]["linear"] is True and profile["x"]["reason"].startswith("autoregressive lag or monotone")


def test_a_sorted_predictor_is_profiled_like_any_other_with_the_option():
    frame = _sorted_curve()
    profile = FeatureProfiler(smooth_monotonic=True).profile(frame, "y", ["x", "z"])
    assert profile["x"]["linear"] is False and profile["x"]["family"] in ("s", "f", "w")


def test_a_time_counter_may_take_a_low_degree_smooth_but_a_lag_stays_linear():
    n = 300
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n, freq="D"), "t": np.arange(n, dtype=float), "y_lag_1": rng.normal(size=n)})
    frame["y"] = 0.01 * frame["t"] ** 1.5 + rng.normal(size=n)
    space = EffectSelector(smooth_monotonic=True).build_search_space(frame, {"targets": ["y"], "features": ["t", "y_lag_1"]}, {"date_col": "ds"})
    assert set(space["t"]["eligible_effects"]) == {"l", "p", "s"} and space["t"]["grids"]["s"]["k"] == [4] and space["t"]["grids"]["p"]["deg"] == [2, 3]
    assert space["y_lag_1"]["eligible_effects"][0] == "l" and "s" not in space["y_lag_1"]["eligible_effects"]
    plain = EffectSelector().build_search_space(frame, {"targets": ["y"], "features": ["t", "y_lag_1"]}, {"date_col": "ds"})
    assert plain["t"]["eligible_effects"] == ["l"]


def test_a_float_column_of_a_small_sample_is_continuous_with_the_option_and_discrete_without():
    values = pd.Series(np.random.default_rng(2).uniform(0, 1, 14), name="v")
    assert EffectSelector()._analyze_topology(values) == "discrete"
    assert EffectSelector(integer_discrete=True)._analyze_topology(values) == "continuous"
    assert EffectSelector(integer_discrete=True)._analyze_topology(pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3.0], name="k")) == "discrete"
    assert EffectSelector(integer_discrete=True)._analyze_topology(pd.Series(list(range(12)), name="id")) == "continuous"      # 12 values in 12 rows


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x, z)", n_experts=3, pop_size=12, use_opera=False, seed=3, export_dir=str(directory), **options)


def test_a_table_sorted_by_its_predictor_finds_the_curve_with_the_options(tmp_path_factory):
    frame = _sorted_curve(400)
    train, test = frame.iloc[::2].copy(), frame.iloc[1::2].copy()
    model = _model(tmp_path_factory.mktemp("g"), smooth_monotonic=True, cross_sectional=True).fit(train, expansions=EXPANSIONS)
    prediction = model.predict(test)[model.trained_experts[0]["name"]].to_numpy()
    r2 = 1.0 - np.sum((prediction - test["y"].to_numpy()) ** 2) / np.sum((test["y"].to_numpy() - test["y"].mean()) ** 2)
    assert r2 > 0.75, r2


def test_the_rows_without_a_date_are_shuffled_before_the_split_with_the_option_only(tmp_path_factory):
    frame = _sorted_curve(300)
    shuffled = _model(tmp_path_factory.mktemp("s"), cross_sectional=True).fit(frame, expansions=EXPANSIONS)
    ordered = _model(tmp_path_factory.mktemp("o")).fit(frame, expansions=EXPANSIONS)
    assert shuffled.ctx.df_val["x"].max() - shuffled.ctx.df_val["x"].min() > 3 and ordered.ctx.df_val["x"].min() > 4


def test_cross_sectional_data_get_no_dynamic_expert_unless_asked(tmp_path_factory):
    frame = _sorted_curve(300)
    model = _model(tmp_path_factory.mktemp("c"), cross_sectional=True).fit(frame)
    assert model.expander.expansions["kalman"] is False and model.expander.expansions["adaptive"] is False
    kept = _model(tmp_path_factory.mktemp("k")).fit(frame, expansions=EXPANSIONS)
    assert kept.expander.expansions["kalman"] is False          # as given
