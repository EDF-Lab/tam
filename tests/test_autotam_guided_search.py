# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Guided search (opt-in): the diagnostics switch on the gene hints (differencing, families), screen the features and write advice.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.diagnostics import feature_diagnostics, residual_advice

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def test_the_screen_finds_a_strong_feature_and_a_noise_feature():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": 3 * x + rng.normal(scale=0.5, size=n), "signal": x, "noise": rng.normal(size=n), "label": ["a"] * n})
    out = feature_diagnostics(frame, "y", ["signal", "noise", "label", "missing", "y"], horizon=1)
    assert set(out) == {"signal", "noise"}
    assert out["signal"]["weak"] is False and abs(out["signal"]["same_step"]) > 0.8
    assert out["noise"]["weak"] is True


def test_the_lagged_correlation_is_taken_inside_each_group():
    rng = np.random.default_rng(1)
    rows = []
    for g in ("a", "b"):
        x = rng.normal(size=300)
        y = np.r_[0.0, x[:-1]] * 2 + rng.normal(scale=0.1, size=300)       # y_t follows x_{t-1}
        rows.append(pd.DataFrame({"g": g, "x": x, "y": y}))
    frame = pd.concat(rows, ignore_index=True)
    out = feature_diagnostics(frame, "y", ["x"], horizon=1, group_col="g")
    assert abs(out["x"]["lagged"]) > 0.8 > abs(out["x"]["same_step"])


def test_the_advice_follows_the_residual_diagnostics():
    assert residual_advice({}) == []
    advice = residual_advice({"structure_left": True, "volatility_clustering": True, "bias_t": 4.2})
    assert len(advice) == 3 and "lags" in advice[0] and "ARCH" in advice[1] and "biased" in advice[2]


def test_the_argument_switches_on_the_hints_and_leaves_what_you_set(tmp_path):
    plain = AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path))
    guided = AutoTAM("y ~ AutoPipe(x1)", guided_search=True, export_dir=str(tmp_path))
    assert (plain.difference, plain.family, plain.diagnostics) == (None, "gaussian", False)
    assert (guided.difference, guided.family, guided.diagnostics) == ("auto", "auto", True)
    kept = AutoTAM("y ~ AutoPipe(x1)", guided_search=True, difference=7, family="gaussian", export_dir=str(tmp_path))
    assert kept.difference == 7 and kept.family == "auto"       # an explicit difference is kept; the family default is a hint


def test_a_guided_fit_reports_its_hints_and_advice(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    x1, junk = rng.normal(size=n), rng.normal(size=n)
    frame = pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n), "x1": x1, "junk": junk,
                          "y": 20 + 3 * x1 + 2 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.4, size=n)})
    model = AutoTAM("y ~ AutoPipe(x1, junk)", n_experts=2, pop_size=6, use_opera=True, seed=2, guided_search=True, export_dir=str(tmp_path))
    model.fit(frame.iloc[:240], date_col="ds", expansions=EXPANSIONS)
    hints = model.guided_hints_
    assert hints["stationarity"] and {"x1", "junk"} <= set(hints["features"]) and "x1" not in hints["weak_features"]
    assert abs(hints["features"]["x1"]["same_step"]) > 3 * abs(hints["features"]["junk"]["same_step"])
    assert isinstance(hints["advice"], list)
    assert model.target_diagnostics_ and model.trained_experts
