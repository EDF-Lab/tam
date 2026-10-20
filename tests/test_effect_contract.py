# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The contract of an effect with the routers of the library: which columns it reads (``required_features``), whether it keeps the trailing
feature dimension (``keeps_feature_dim``), what it says about itself in ``summary()`` (``summary_details``), and the initialisation of the
effects that hold data-dependent state (neural weights of ``phys(basis='neural')``, wavelet grid, centres, projections) before any penalty is built.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.spectrum import (CategoricalEffect, NeuralEffect, RBFEffect, SplineEffect, TensorProductEffect, TreeEffect, WaveletEffect,
                                UniversalPhysicsEffect)


def _curve(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.uniform(0, 1, n), "y": np.sin(4 * x) + 0.1 * rng.normal(size=n)})


def test_a_physics_effect_on_a_neural_basis_fits():
    frame = _curve()
    model = ta.StaticTAM("y ~ phys(x, basis='neural', k=20)").fit(frame)
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"].to_numpy() - frame["y"].to_numpy()) ** 2)) < 0.2


def test_the_wavelet_grid_and_the_physics_basis_come_from_the_whole_training_tensor():
    frame = _curve()
    model = ta.StaticTAM("y ~ w(x, n_scales=3, n_locations=6) + phys(z, basis='spline', k=10)").fit(frame)
    wavelet = next(e for e in model.effects_list_ if isinstance(e, WaveletEffect))
    assert wavelet.locations is not None and float(wavelet.locations.min()) <= -0.99 and float(wavelet.locations.max()) >= 0.99


def test_the_columns_an_effect_reads():
    assert SplineEffect("x", 6, 3, 2, 1e-3, "linear").required_features == ["x"]
    assert RBFEffect("x", 10, None, None, 1e-3, ["z"], "continue").required_features == ["x", "z"]
    graph = CategoricalEffect("region", 9, "graph", 1e-3, 1, "continue", coords=["x", "y"])
    assert graph.required_features == ["region", "x", "y"] and graph.multi_input
    te = TensorProductEffect([SplineEffect("x", 6, 3, 2, 1e-3, "linear"), RBFEffect("a", 10, None, None, 1e-3, ["b"], "continue")], 1.0, "linear")
    assert te.required_features == ["x", "a", "b"] and te.keeps_feature_dim
    assert RBFEffect("x", 10, None, None, 1e-3, None, "continue").keeps_feature_dim
    assert NeuralEffect("x", 10, "relu", 1e-3, None, 0, 1, "linear").keeps_feature_dim
    assert not SplineEffect("x", 6, 3, 2, 1e-3, "linear").keeps_feature_dim
    physics = UniversalPhysicsEffect("x", "neural", 10, {"D2": 1.0}, 1e-3, "continue")
    assert physics.keeps_feature_dim and physics.required_features == ["x"]


def test_summary_is_written_by_the_effects():
    frame = _curve()
    frame["g"] = (frame["x"] * 5).astype(int)
    model = ta.StaticTAM("y ~ s(x, k=8) + l(z) + c(g, ap=-3) + rbf(x, others='z', kernel='tps', n_centers=15) + t(x, n_trees=5, max_depth=2)").fit(frame)
    table = model.summary()
    details = dict(zip(table["Type"], table["Structure / Params"]))
    assert "Knots=8" in details["Spline"] and details["Linear"] == "Slope (Ridge)" and "Unseen=edge" in details["Categorical"]
    assert details["RBF"].startswith("TPS(m=2)") and "Trees=5" in details["Tree"]
    assert details["Offset"] == "Global Bias"


def test_decomposition_names_know_the_new_kernels_and_topologies():
    from tam.model._math import decomposition_names
    rng = np.random.default_rng(1)
    frame = _curve()
    frame["region"] = rng.integers(0, 6, len(frame))
    model = ta.StaticTAM("y ~ rbf(x, others='z', kernel='tps', n_centers=15) + s(x, k=6) + c(region, n_cat=6, topo='graph', coords='x:z', ap=-3) + c(region, n_cat=6, ap=-3)").fit(frame)
    names = decomposition_names(model.effects_list_)
    assert all("categorical" not in n and "rbf_tps" not in n for n in names), names


def test_missing_levels_of_a_graph_term_only_warn():
    rng = np.random.default_rng(2)
    frame = _curve()
    frame["region"] = rng.integers(0, 4, len(frame))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ta.StaticTAM("y ~ c(region, n_cat=8, topo='graph', coords='x:z', ap=-3)").fit(frame)
    assert any("missing" in str(w.message) for w in caught)


def test_a_zero_weight_of_an_adaptive_penalty_raises_no_warning():
    effect = SplineEffect("x", 12, 3, 2, 1e-3, "linear", m_pen=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        effect.set_penalty_coordinates([1.0, 0.0, 1.0])
    assert np.isfinite(effect.lambda_p)


def test_the_adaptive_grid_search_initialises_the_candidate_effects():
    rng = np.random.default_rng(3)
    n = 600
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)})
    frame["y"] = 1.0 + frame["x"] + np.sin(5 * frame["z"]) * (np.arange(n) > 300) + 0.1 * rng.normal(size=n)
    base = ta.StaticTAM("y ~ l(x)", date_col="date").fit(frame.iloc[:300])
    model = ta.AdaptiveTAM(base_model=base, adaptive_formula="Residualy ~ rbf(x, others='z', kernel='tps', n_centers=15, ap='pen')",
                           update_interval_periods=10, training_window_periods=60, steps_per_period=1, date_col="date")
    best = model.grid_search_fit(frame, {"pen": [-6.0, -2.0]})
    effect = next(e for e in best.adaptive_model_.effects_list_ if isinstance(e, RBFEffect))
    assert effect.centers is not None
