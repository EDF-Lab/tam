# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``difference="select"``: a level model and a differenced model are fitted on the earlier rows, forecast the last three horizons, and the one
with the smaller error in the level scale is kept.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _cumulative(n=260, seed=4):
    rng = np.random.default_rng(seed)
    step = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": rng.normal(size=n), "y": 100 + np.cumsum(step)})


def _stationary(n=260, seed=5):
    """No trend and no season: the level is the right thing to model (a lag-7 difference of a weekly series is a strong model by itself)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": x1, "y": 30 + 2 * x1 + rng.normal(scale=0.4, size=n)})


def _model(tmp_path, **kwargs):
    return AutoTAM("y ~ AutoPipe(x1)", lags=[7], n_experts=1, pop_size=4, use_opera=False, seed=3, export_dir=str(tmp_path), **kwargs)


def test_a_cumulative_series_keeps_the_difference_and_the_scores_say_why(tmp_path):
    model = _model(tmp_path, difference="select").fit(_cumulative(), date_col="ds", expansions=EXPANSIONS)
    assert model.difference_ == 7
    assert model.difference_scores_["difference"] < model.difference_scores_["level"]


def test_a_stationary_series_keeps_the_level(tmp_path):
    model = _model(tmp_path, difference="select").fit(_stationary(), date_col="ds", expansions=EXPANSIONS)
    assert model.difference_ is None
    assert model.difference_scores_["level"] <= model.difference_scores_["difference"]


def test_a_history_too_short_to_hold_out_three_horizons_keeps_the_level_without_an_error(tmp_path):
    model = _model(tmp_path, difference="select").fit(_stationary().iloc[:70], date_col="ds", expansions=EXPANSIONS)
    assert model.difference_ is None and model.difference_scores_ == {}


def test_the_selection_needs_a_date_column(tmp_path):
    with pytest.raises(ValueError, match="date column"):
        _model(tmp_path, difference="select").fit(_stationary().drop(columns="ds"), expansions=EXPANSIONS)


def test_a_seasonal_series_may_keep_the_difference_when_the_lag_is_its_season(tmp_path):
    """A weekly series forecast 7 steps ahead: the lag-7 difference removes the season, so it is a legitimate winner, not a mistake."""
    rng = np.random.default_rng(6)
    n = 260
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "x1": x1,
                          "y": 30 + 4 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * x1 + rng.normal(scale=0.4, size=n)})
    model = _model(tmp_path, difference="select").fit(frame, date_col="ds", expansions=EXPANSIONS)
    assert set(model.difference_scores_) == {"level", "difference"} and model.difference_ in (None, 7)
