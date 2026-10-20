# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Validation folds: ``rolling_folds`` with a one-step horizon (or none) must not give folds of one row, which the search skips, and a long daily or
monthly series has a bounded number of expanding-window folds (``max_cv_folds``, 5 with ``search="fast"``).
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.data_manager import DataManager

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=300, freq="D", seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 10 + 2 * x + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2018-01-01", periods=n, freq=freq), "x1": x, "y": y})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=False, seed=2, export_dir=str(directory), **options)


@pytest.mark.parametrize("horizon", [None, 1])
def test_rolling_folds_without_a_horizon_or_with_a_one_step_horizon_keep_a_scored_search(tmp_path, horizon):
    model = _model(tmp_path, rolling_folds=3, horizon=horizon).fit(_frame(), date_col="ds", expansions=EXPANSIONS)
    assert model.architectures_ and any(model.architectures_.values())
    assert all(len(fold[1]) >= 2 for fold in model.ctx.cv_folds)
    assert model.trained_experts


def test_a_horizon_of_several_steps_keeps_its_folds(tmp_path):
    model = _model(tmp_path, rolling_folds=3, horizon=7, lags=[7]).fit(_frame(), date_col="ds", expansions=EXPANSIONS)
    assert [len(fold[1]) for fold in model.ctx.cv_folds] == [7, 7, 7]


def _folds(n, freq, cap):
    manager = DataManager("y ~ AutoPipe(x1)", max_cv_folds=cap)
    frame = _frame(n, freq)
    split = int(n * 0.8)
    return manager.generate_cv_folds(frame.iloc[:split], frame.iloc[split:], "ds", "auto")


def test_a_long_daily_series_has_a_fold_every_thirty_rows_without_a_cap_and_at_most_the_cap_with_one():
    assert len(_folds(2400, "D", None)) > 30
    assert len(_folds(2400, "D", 10)) <= 11                    # the last fold takes the remainder
    assert len(_folds(2400, "D", 5)) <= 6


def test_a_long_monthly_series_is_capped_too():
    assert len(_folds(240, "MS", None)) > 8
    assert len(_folds(240, "MS", 5)) <= 6


def test_a_short_series_is_not_changed_by_a_cap():
    assert len(_folds(200, "D", 10)) == len(_folds(200, "D", None))


def test_the_fast_preset_caps_the_folds_and_a_given_value_wins(tmp_path):
    assert _model(tmp_path, search="fast").data_manager.max_cv_folds == 5
    assert _model(tmp_path, search="fast", max_cv_folds=8).data_manager.max_cv_folds == 8
    assert _model(tmp_path).data_manager.max_cv_folds is None


@pytest.mark.parametrize("bad", [0, -2, 2.5, True])
def test_a_cap_that_is_not_a_positive_whole_number_is_refused(bad):
    with pytest.raises(ValueError, match="max_cv_folds"):
        DataManager("y ~ AutoPipe(x1)", max_cv_folds=bad)
