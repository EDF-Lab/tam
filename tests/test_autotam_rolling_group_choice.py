# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Choices on the rolling folds: ``group_search_by="rolling"`` compares the grouped and the ungrouped fit on their rolling folds, which cover the same
days, instead of the validation rows.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _hourly(days=100, seed=3):
    rng = np.random.default_rng(seed)
    n = 24 * days
    hour = np.arange(n) % 24
    x1 = rng.normal(size=n)
    slope = 1.0 + 4.0 * np.sin(2 * np.pi * hour / 24) ** 2 * np.where(hour < 12, 1.0, -1.0)
    y = 50 + slope * x1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "x1": x1, "y": y})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, lags=[24], export_dir=str(directory), **options)


def test_a_rolling_block_of_a_derived_group_is_one_cycle_of_every_group(tmp_path):
    model = _model(tmp_path, group_search="always", rolling_folds=3).fit(_hourly(), date_col="ds", expansions=EXPANSIONS)
    folds = model.ctx.cv_folds
    assert len(folds) == 3
    assert all(len(fold[1]) == 24 and fold[1][model.ctx.group_col].nunique() == 24 for fold in folds)   # one day: one row in each of the 24 groups


def test_the_two_versions_are_compared_on_the_rolling_folds(tmp_path):
    model = _model(tmp_path, group_search=True, group_search_by="rolling", rolling_folds=3).fit(_hourly(), date_col="ds", expansions=EXPANSIONS)
    assert model.group_choice_ == "rolling" and set(model.group_scores_) == {"ungrouped", "time_of_day"}
    assert (model.derived_group_ is not None) == (model.group_scores_["time_of_day"] < model.group_scores_["ungrouped"])


def test_the_validation_rows_decide_by_default(tmp_path):
    model = _model(tmp_path, group_search=True, rolling_folds=3).fit(_hourly(), date_col="ds", expansions=EXPANSIONS)
    assert model.group_choice_ == "validation" and set(model.group_scores_) == {"ungrouped", "time_of_day"}


def test_without_rolling_folds_the_rolling_choice_says_so_and_uses_the_validation_rows(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam"):
        model = _model(tmp_path, group_search=True, group_search_by="rolling").fit(_hourly(), date_col="ds", expansions=EXPANSIONS)
    assert model.group_choice_ == "validation"
    assert any("rolling_folds" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_an_unknown_way_to_choose_is_refused(tmp_path):
    with pytest.raises(ValueError, match="group_search_by"):
        _model(tmp_path, group_search=True, group_search_by="oracle")
