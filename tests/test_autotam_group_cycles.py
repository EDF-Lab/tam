# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Group search counts in cycles: inside a derived group one step is one cycle (a day for an hourly series), so the lags of the seed formulas, the
past-covariate orders and the season the caller gives in rows are carried over as whole cycles, and the adaptive windows are bounded by the cycles
of one group. What cannot be carried over is said at warning level.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.group_search import GROUP_COLUMN, covariates_in_cycles, formula_in_cycles, to_cycles

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": True, "grid": False}


def test_rows_are_cycles_only_when_they_make_whole_cycles():
    assert to_cycles(168, 24) == 7 and to_cycles(24, 24) == 1
    assert to_cycles(25, 24) is None and to_cycles(12, 24) is None and to_cycles(0, 24) is None


def test_the_lags_of_a_formula_are_written_in_cycles():
    assert formula_in_cycles("y ~ l(y_lag_168) + l(y_lag_24) + l(x1)", 24) == "y ~ l(y_lag_7) + l(y_lag_1) + l(x1)"
    assert formula_in_cycles("y ~ l(x1)", 24) == "y ~ l(x1)"
    assert formula_in_cycles("y ~ l(y_lag_25)", 24) is None


def test_the_orders_of_the_covariates_are_written_in_cycles_and_the_others_dropped():
    assert covariates_in_cycles({"x": [24, 48, 30], "z": [5]}, 24) == {"x": [1, 2]}


def _hourly(days=110, seed=3):
    rng = np.random.default_rng(seed)
    n = 24 * days
    hour = np.arange(n) % 24
    x1 = rng.normal(size=n)
    slope = 1.0 + 4.0 * np.sin(2 * np.pi * hour / 24) ** 2 * np.where(hour < 12, 1.0, -1.0)
    y = 50 + slope * x1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "x1": x1, "y": y})


def _model(directory, lags=(24,), n_experts=2, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=n_experts, pop_size=4, use_opera=True, seed=5, group_search="always", lags=list(lags), export_dir=str(directory), **options)


@pytest.mark.slow
def test_a_seed_formula_with_a_lag_in_rows_is_carried_over_to_the_grouped_fit(tmp_path):
    frame = _hourly(days=65)
    model = _model(tmp_path).fit(frame.iloc[:24 * 60], date_col="ds", expansions={**EXPANSIONS, "adaptive": False}, seed_formulas=["y ~ l(x1) + l(y_lag_24)"])
    assert model.derived_group_ is not None and model.ctx.group_col == GROUP_COLUMN
    assert any(name.startswith("Static_Seed_R1") for name in model.experts_["name"])
    assert any("y_lag_1" in str(f) for f in model.architectures_["Seed"])


def test_a_seed_formula_that_cannot_be_carried_over_keeps_the_ungrouped_fit_and_says_so(tmp_path, caplog):
    frame = _hourly()
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam"):
        model = _model(tmp_path, lags=(24, 25)).fit(frame.iloc[:24 * 90], date_col="ds", expansions=EXPANSIONS, seed_formulas=["y ~ l(x1) + l(y_lag_25)"])
    assert model.derived_group_ is None
    assert any("seed formula" in r.message and r.levelno == logging.WARNING for r in caplog.records)
    assert any("ungrouped fit is kept" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_a_season_in_rows_that_is_not_whole_cycles_keeps_the_ungrouped_fit_and_says_so(tmp_path, caplog):
    frame = _hourly()
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam"):
        model = _model(tmp_path, season=30).fit(frame.iloc[:24 * 90], date_col="ds", expansions=EXPANSIONS)
    assert model.derived_group_ is None
    assert any("season" in r.message and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.slow
def test_the_adaptive_windows_of_a_derived_group_fit_in_the_cycles_of_one_group(tmp_path, caplog):
    frame = _hourly(days=66)
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam"):
        model = _model(tmp_path, n_experts=1, online_variants=True).fit(frame.iloc[:24 * 62], date_col="ds", expansions=EXPANSIONS)
    cycles = 62                                   # days in the fit, dev and validation rows: one row a day in each group
    assert model.derived_group_ is not None
    assert model.ctx.candidate_windows and max(model.ctx.candidate_windows) <= (cycles * 2 // 3) // 2
    assert any("skipped" in r.message for r in caplog.records)


def test_an_ungrouped_fit_keeps_its_windows(tmp_path):
    frame = _hourly(days=60)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, lags=[24], export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert model.ctx.cycle_windows is False
