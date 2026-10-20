# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Regimes detected in the training target: a level shift is found, white noise gives none, a forecast row carries the regime in force at the
cutoff, and the regime is a category for the search.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import regimes as rg
from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _series(n=400, drop_at=250, drop=-6.0, seed=3):
    rng = np.random.default_rng(seed)
    y = 50 + rng.normal(size=n)
    y[drop_at:drop_at + 90] += drop
    return y


def test_a_three_month_level_drop_is_found_near_its_start_and_its_end():
    breaks = rg.detect_breaks(_series())
    assert len(breaks) == 2 and abs(breaks[0] - 250) <= 6 and abs(breaks[1] - 340) <= 6


def test_white_noise_and_a_pure_season_give_no_break():
    assert rg.detect_breaks(np.random.default_rng(0).normal(size=500)) == []
    t = np.arange(24 * 30)
    seasonal = 10 + 4 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(1).normal(scale=0.3, size=len(t))
    assert rg.detect_breaks(seasonal, season=24) == []


def test_the_regime_of_a_date_after_the_last_break_is_the_last_regime():
    dates = pd.Series(pd.date_range("2024-01-01", periods=10))
    ids = rg.regime_ids(dates, [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-08")])
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    later = rg.regime_ids(pd.Series(pd.date_range("2025-06-01", periods=3)), [pd.Timestamp("2024-01-04")])
    assert later.tolist() == [1, 1, 1]


def test_no_regime_of_the_training_rows_depends_on_data_after_the_cutoff():
    y = _series()
    cut = y[:300]
    dates = pd.Series(pd.date_range("2023-01-01", periods=400))
    short = rg.fit_regimes(pd.DataFrame({"ds": dates[:300], "y": cut}), "y", "ds", None, None)
    longer = rg.fit_regimes(pd.DataFrame({"ds": dates, "y": y}), "y", "ds", None, None)
    assert len(short["breaks"][None]) == 1                          # the drop has begun and has not ended at the cutoff
    assert short["breaks"][None][0] == longer["breaks"][None][0]
    forecast = rg.regime_ids(dates[300:305], short["breaks"][None])
    assert (forecast == 1).all()


def test_each_group_has_its_own_breaks():
    frame = pd.concat([pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=400), "g": 0, "y": _series()}),
                       pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=400), "g": 1, "y": _series(drop=0.0, seed=4)})])
    fitted = rg.fit_regimes(frame, "y", "ds", "g", None)
    assert len(fitted["breaks"][0]) == 2 and fitted["breaks"][1] == []
    out = frame.copy()
    rg.add_regime_column(out, "ds", "g", fitted)
    assert out.loc[out["g"] == 1, "regime"].eq(0).all() and out.loc[out["g"] == 0, "regime"].max() == 2


def _frame(n=420):
    rng = np.random.default_rng(6)
    y = _series(n, 280)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="D"), "x1": x1, "y": y + 0.5 * x1})


def test_the_search_gets_the_regime_as_a_category_and_the_forecast_rows_carry_the_last_regime(tmp_path):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=2, regimes=True, export_dir=str(tmp_path))
    model.fit(frame.iloc[:400], date_col="ds", expansions=EXPANSIONS)
    assert "regime" in model.ctx.df_all_aug.columns and model.ctx.regime["breaks"][None]
    assert set(model.ctx.search_space["regime"]["eligible_effects"]) == {"c"}
    out = model.predict(frame.iloc[400:405], date_col="ds")
    assert len(out) == 5 and np.isfinite(out.iloc[:, 0]).all()


def test_without_the_argument_or_without_a_break_nothing_is_added(tmp_path):
    frame = _frame()
    plain = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=2, export_dir=str(tmp_path)).fit(
        frame.iloc[:300], date_col="ds", expansions=EXPANSIONS)
    assert "regime" not in plain.ctx.df_all_aug.columns and not plain.ctx.regime
    calm = frame.iloc[:200].assign(y=lambda d: 50 + 0.5 * d["x1"] + np.random.default_rng(1).normal(size=len(d)))
    none = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=2, regimes=True, export_dir=str(tmp_path)).fit(
        calm, date_col="ds", expansions=EXPANSIONS)
    assert "regime" not in none.ctx.df_all_aug.columns and not none.ctx.regime


def test_regimes_need_a_date_column(tmp_path):
    with pytest.raises(ValueError, match="date column"):
        AutoTAM("y ~ AutoPipe(x1)", regimes=True, export_dir=str(tmp_path)).fit(_frame().iloc[:200].drop(columns="ds"))


def test_no_regime_is_shorter_than_four_seasons():
    y = _series(400, drop_at=380, drop=-6.0)          # a drop in the last 20 rows
    breaks = rg.detect_breaks(y, season=7)
    assert all(28 <= b <= len(y) - 28 for b in breaks)   # a regime lasts at least four weeks of 7 steps: no break leaves a shorter one
    assert len(rg.detect_breaks(_series(400, drop_at=300, drop=-6.0), season=7)) >= 1


def test_forecast_rows_carry_the_previous_regime_while_the_last_one_is_short():
    n = 420
    y = _series(n, drop_at=360, drop=-8.0)
    y[360:] = y[360:] - y[360:].mean() + (y[:360].mean() - 8.0)      # the last 60 rows stay at the lower level
    dates = pd.date_range("2023-01-01", periods=n)
    fitted = rg.fit_regimes(pd.DataFrame({"ds": dates, "y": y}), "y", "ds", None, 7)
    assert len(fitted["breaks"][None]) == 1                           # the lower level lasted 60 rows (about 8.5 weeks): trusted
    short = _series(n, drop_at=380, drop=-8.0)
    short[380:] = short[380:] - short[380:].mean() + (short[:380].mean() - 8.0)
    fitted_short = rg.fit_regimes(pd.DataFrame({"ds": dates, "y": short}), "y", "ds", None, 7)
    assert len(fitted_short["breaks"][None]) == 1 and fitted_short["carry"][None] == 0     # 40 rows: below eight weeks, not trusted yet
    frame = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=5)})
    rg.add_regime_column(frame, "ds", None, fitted_short)
    assert frame["regime"].tolist() == [0.0] * 5
    inside = pd.DataFrame({"ds": dates[-3:]})
    rg.add_regime_column(inside, "ds", None, fitted_short)
    assert inside["regime"].tolist() == [1.0] * 3                      # the training rows keep their own regime
    ahead_trusted = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=2)})
    rg.add_regime_column(ahead_trusted, "ds", None, fitted)
    assert ahead_trusted["regime"].tolist() == [1.0, 1.0]
