# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Statistical diagnostics of a target and of the residuals of the final model.
"""

import numpy as np
import pandas as pd

from tam.model.autotam import diagnostics as dg
from tam.model.autotam.auto_tam import AutoTAM

RNG = np.random.default_rng(0)


def test_a_random_walk_has_a_unit_root_and_white_noise_is_stationary():
    assert dg.stationarity_verdict(np.cumsum(np.random.default_rng(0).normal(size=400)))["verdict"] == "unit_root"
    assert dg.stationarity_verdict(np.random.default_rng(0).normal(size=400))["verdict"] == "stationary"


def test_a_series_that_follows_a_linear_trend_is_not_called_a_unit_root():
    t = np.arange(400)
    assert dg.stationarity_verdict(0.05 * t + np.random.default_rng(0).normal(size=400))["verdict"] == "trend_stationary"


def test_a_seasonal_series_has_a_strong_seasonal_part_and_noise_does_not():
    t = np.arange(24 * 40)
    seasonal = 10 + 3 * np.sin(2 * np.pi * t / 24) + RNG.normal(scale=0.5, size=len(t))
    assert dg.seasonal_strength(seasonal, 24)["seasonal"] > 0.6
    assert dg.seasonal_strength(RNG.normal(size=len(t)), 24)["seasonal"] < 0.2


def test_ljung_box_rejects_the_residuals_of_an_ar1_fitted_by_a_constant_and_not_the_true_innovations():
    e = RNG.normal(size=600)
    y = np.zeros(600)
    for i in range(1, 600):
        y[i] = 0.8 * y[i - 1] + e[i]
    assert dg.ljung_box(y - y.mean(), 10)["p_value"] < 0.01
    assert dg.ljung_box(e, 10)["p_value"] > 0.01
    assert dg.residual_diagnostics(y - y.mean())["structure_left"] is True
    assert dg.residual_diagnostics(e)["structure_left"] is False


def test_arch_residuals_are_flagged():
    e = np.zeros(800)
    z = RNG.normal(size=800)
    for i in range(1, 800):
        e[i] = np.sqrt(0.2 + 0.7 * e[i - 1] ** 2) * z[i]
    assert dg.residual_diagnostics(e)["volatility_clustering"] is True


def test_the_demand_class_and_outliers_and_breaks():
    lumpy = np.where(RNG.random(300) < 0.2, RNG.lognormal(1.0, 1.5, 300), 0.0)
    assert dg.demand_class(lumpy) in ("lumpy", "intermittent")
    assert dg.demand_class(RNG.normal(10, 1, 100)) is None
    step = np.r_[RNG.normal(0, 1, 150), RNG.normal(4, 1, 150)]
    assert dg.cusum_break(step)["break"] is True and dg.cusum_break(RNG.normal(size=300))["break"] is False
    spiky = RNG.normal(0, 1, 300)
    spiky[::30] = 40
    assert dg.target_diagnostics(spiky)["outlier_share"] > 0.02


def test_too_few_rows_give_only_the_count():
    assert dg.target_diagnostics(np.arange(10.0)) == {"n": 10}
    assert dg.residual_diagnostics(np.arange(10.0)) == {"n": 10}


def test_the_fit_keeps_the_diagnostics_only_when_asked_and_reads_only_the_training_rows(tmp_path):
    n = 260
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": RNG.normal(size=n)})
    frame["y"] = 5 + 2 * frame["x1"] + RNG.normal(scale=0.3, size=n)
    cut = frame.iloc[:200]
    expansions = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}
    on = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=1, diagnostics=True, export_dir=str(tmp_path)).fit(
        cut, date_col="ds", expansions=expansions)
    assert on.target_diagnostics_["n"] == 200 and on.residual_profile_["n"] > 0 and "ljung_box_p" in on.residual_profile_
    off = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=1, export_dir=str(tmp_path)).fit(cut, date_col="ds", expansions=expansions)
    assert off.target_diagnostics_ == {} and off.residual_profile_ == {}
