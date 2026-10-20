# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Selection on the weighted quantile loss: the loss of the quantile layer (point forecast plus the quantiles of its errors), scale-free.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.pipeline.ensemble_selector import EnsembleSelector
from tam.model.autotam.pipeline.expert_expander import ExpertExpander
from tam.model.autotam.quantile_loss import DEFAULT_LEVELS, is_wql, parse_levels, wql

RNG = np.random.default_rng(0)


def test_a_perfect_forecast_has_no_loss_and_a_noisier_one_has_more():
    y = 50 + RNG.normal(size=400)
    assert wql(y, y) == 0.0
    small, large = y + RNG.normal(scale=0.2, size=400), y + RNG.normal(scale=2.0, size=400)
    assert 0.0 < wql(y, small) < wql(y, large)


def test_the_loss_does_not_depend_on_the_scale_of_the_series():
    y = 50 + RNG.normal(size=300)
    forecast = y + RNG.normal(scale=0.5, size=300)
    assert wql(10 * y, 10 * forecast) == pytest.approx(wql(y, forecast))


def test_a_biased_forecast_is_not_penalised_for_its_constant_bias():
    y = 50 + RNG.normal(size=500)
    forecast = y + RNG.normal(scale=0.5, size=500)
    assert wql(y, forecast + 3.0) == pytest.approx(wql(y, forecast), rel=1e-9)      # the quantile layer re-centres on the errors


def test_the_levels_come_from_the_metric_name():
    assert is_wql("WQL") and is_wql("wql:0.05,0.5") and not is_wql("rmse")
    assert parse_levels("wql") == DEFAULT_LEVELS and parse_levels("wql:0.9,0.1") == (0.1, 0.9)
    for bad in ("wql:", "wql:1.2", "wql:0,0.5", "wql:a,b"):
        with pytest.raises(ValueError):
            parse_levels(bad)


def test_the_three_metric_functions_know_wql():
    y = 20 + RNG.normal(size=100)
    forecast = y + RNG.normal(scale=0.7, size=100)
    expected = wql(y, forecast, parse_levels("wql:0.1,0.9"))
    for scorer in (ExpertExpander()._calculate_error, EnsembleSelector()._calculate_error, DragTAM(target_col="y", population_size=4)._calculate_error):
        assert scorer(y, forecast, "wql:0.1,0.9") == pytest.approx(expected)
        assert scorer(y, forecast, "rmse") == pytest.approx(np.sqrt(np.mean((y - forecast) ** 2)))     # the others are unchanged


def test_a_fit_selected_on_the_quantile_loss_gives_a_final_model_and_quantiles(tmp_path):
    n = 220
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": RNG.normal(size=n)})
    frame["y"] = 20 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * frame["x1"] + RNG.normal(scale=0.4, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=2, export_dir=str(tmp_path))
    model.fit(frame.iloc[:180], date_col="ds", optimization_metric="wql", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    out = model.predict_quantiles(frame.iloc[180:190], taus=(0.1, 0.9), date_col="ds")
    assert model.ctx.optimization_metric == "wql" and (out["q0.1"] <= out["q0.9"]).all()
