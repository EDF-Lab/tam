# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Search on a sample of the groups, then fit the architectures found on every group.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _panel(n_groups=24, n=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x1 = rng.normal(size=n)
        y = 10 * (g + 1) + 3 * x1 + rng.normal(scale=0.5, size=n)
        rows.append(pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "g": g, "x1": x1, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _model(tmp_path, **kwargs):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=8, export_dir=str(tmp_path), **kwargs)


def test_the_search_runs_on_an_even_sample_of_the_groups_by_level_and_every_group_is_fitted(tmp_path):
    panel = _panel()
    model = _model(tmp_path, search_groups=6).fit(panel, date_col="ds", group_col="g", expansions=EXPANSIONS)
    assert len(model.search_sample_) == 6 and model.search_sample_[0] == 0 and model.search_sample_[-1] == 23
    assert set(model.ctx.df_all_aug["g"]) == set(range(24))
    assert model.architectures_ and model.trained_experts


def test_the_sample_is_the_same_on_every_run(tmp_path):
    panel = _panel()
    first = _model(tmp_path, search_groups=5).fit(panel, date_col="ds", group_col="g", expansions=EXPANSIONS)
    second = _model(tmp_path, search_groups=5).fit(panel, date_col="ds", group_col="g", expansions=EXPANSIONS)
    assert first.search_sample_ == second.search_sample_ and first.architectures_ == second.architectures_


def test_without_the_argument_or_with_few_groups_the_search_is_on_everything(tmp_path):
    panel = _panel(n_groups=4)
    assert _model(tmp_path).fit(panel, date_col="ds", group_col="g", expansions=EXPANSIONS).search_sample_ == []
    assert _model(tmp_path, search_groups=10).fit(panel, date_col="ds", group_col="g", expansions=EXPANSIONS).search_sample_ == []


def test_search_groups_has_no_effect_without_a_group_column(tmp_path):
    model = _model(tmp_path, search_groups=3).fit(_panel(1, 120).drop(columns="g"), date_col="ds", expansions=EXPANSIONS)
    assert model.search_sample_ == []
