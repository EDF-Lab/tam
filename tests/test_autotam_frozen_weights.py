# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The Apex quality floor and the frozen weights of ``predict()`` are arguments of AutoTAM; their defaults are the search as it was.
"""

import types

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.ensemble_selector import EnsembleSelector


def test_the_defaults_are_the_search_as_it_was(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1)", export_dir=str(tmp_path))
    assert model.selector.apex_quality_ratio == 1.5 and model.selector.weight_average == "snapshot"


def test_the_arguments_reach_the_selector(tmp_path):
    model = AutoTAM("y ~ AutoPipe(x1)", apex_quality_ratio=2.0, frozen_weights="mean_30", export_dir=str(tmp_path))
    assert model.selector.apex_quality_ratio == 2.0 and model.selector.weight_average == "mean_30"
    assert AutoTAM("y ~ AutoPipe(x1)", apex_quality_ratio=None, export_dir=str(tmp_path)).selector.apex_quality_ratio is None


def test_bad_values_are_refused(tmp_path):
    with pytest.raises(ValueError, match="apex_quality_ratio"):
        AutoTAM("y ~ AutoPipe(x1)", apex_quality_ratio=0.5, export_dir=str(tmp_path))
    with pytest.raises(ValueError, match="weight_average"):
        AutoTAM("y ~ AutoPipe(x1)", frozen_weights="median", export_dir=str(tmp_path))


def test_each_average_reads_the_right_part_of_the_trajectory():
    trajectory = pd.Series(np.arange(1.0, 101.0))            # 100 validation steps, weights 1..100
    ctx = types.SimpleNamespace(steps_per_period=1)
    pick = lambda mode: EnsembleSelector(weight_average=mode)._frozen_weight(trajectory, ctx)
    assert pick("snapshot") == 100.0
    assert pick("mean_7") == np.mean(np.arange(94.0, 101.0))
    assert pick("mean_30") == np.mean(np.arange(71.0, 101.0))
    assert pick("full") == 50.5


def test_the_window_is_in_days_so_it_follows_the_sampling_step():
    trajectory = pd.Series(np.arange(1.0, 401.0))
    hourly = types.SimpleNamespace(steps_per_period=24)
    assert EnsembleSelector(weight_average="mean_7")._frozen_weight(trajectory, hourly) == np.mean(np.arange(233.0, 401.0))   # 168 steps
    short = pd.Series([2.0, 4.0])
    assert EnsembleSelector(weight_average="mean_30")._frozen_weight(short, types.SimpleNamespace(steps_per_period=1)) == 3.0


def test_a_fit_with_a_floor_and_an_average_still_gives_a_final_model(tmp_path):
    rng = np.random.default_rng(2)
    n = 260
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    frame["y"] = 5 + 2 * frame["x1"] + rng.normal(scale=0.3, size=n)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=1, apex_quality_ratio=2.0, frozen_weights="mean_7",
                    export_dir=str(tmp_path))
    model.fit(frame.iloc[:220], date_col="ds", expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False})
    out = model.predict(frame.iloc[220:225], date_col="ds")
    assert model.final_model_column_ in out.columns and np.isfinite(out[model.final_model_column_]).all()
