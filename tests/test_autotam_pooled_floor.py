# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The pooled seasonal floor (opt-in): a seasonal profile always in the pool, and the final model never worse than it on the validation rows.
"""

import types

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.floor import FLOOR_COLUMN, FLOOR_ISLAND, floor_formula, season_position
from tam.model.autotam.pipeline.ensemble_selector import EnsembleSelector


@pytest.mark.parametrize("freq, steps, count, first", [("h", 24.0, 24, 0), ("D", 1.0, 7, 0), ("W", 1 / 7, 52, None), ("MS", 1 / 30.4, 12, 0), ("QS", 1 / 91.3, 4, 0)])
def test_the_position_follows_the_natural_season_of_the_step(freq, steps, count, first):
    dates = pd.Series(pd.date_range("2024-01-01", periods=120, freq=freq))
    positions, k = season_position(dates, steps)
    assert k == count and positions.min() >= 0 and positions.max() < count
    if first is not None:
        assert positions[0] == first


def test_the_hours_and_the_weekdays_are_the_positions_of_hourly_and_daily_data():
    hourly = pd.Series(pd.date_range("2024-01-01 00:00", periods=48, freq="h"))
    assert season_position(hourly, 24.0)[0].tolist() == list(range(24)) * 2
    daily = pd.Series(pd.date_range("2024-01-01", periods=14, freq="D"))                 # a Monday
    assert season_position(daily, 1.0)[0].tolist() == list(range(7)) * 2


def test_a_step_without_a_natural_season_is_refused():
    with pytest.raises(ValueError, match="natural season"):
        season_position(pd.Series(pd.date_range("2024-01-01", periods=10, freq="3D")), 1 / 3)
    with pytest.raises(ValueError, match="regular"):
        season_position(pd.Series(pd.date_range("2024-01-01", periods=3)), None)


def test_the_formula_has_one_coefficient_per_position():
    assert floor_formula("y", 24) == f"y ~ c({FLOOR_COLUMN}, n_cat=24, topo='nominal')"


def _validation(apex_noise, floor_noise, n=60):
    rng = np.random.default_rng(0)
    y = 10 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "Floor_one": y + rng.normal(scale=floor_noise, size=n), "Other": y + rng.normal(scale=2.0, size=n),
                         "AutoTAM_Apex_Ensemble": y + rng.normal(scale=apex_noise, size=n)})


def test_a_floor_better_than_the_apex_becomes_the_final_model_and_a_seed_is_not_a_floor_unless_asked():
    experts = [{"name": "Floor_one", "island": FLOOR_ISLAND}, {"name": "Other", "island": "LinearIsland"}]
    weights = {"Other": 1.0}
    frame = _validation(apex_noise=1.5, floor_noise=0.1)
    ctx = types.SimpleNamespace(target="y")
    assert EnsembleSelector(pooled_floor=True)._apply_seed_floor(weights, experts, frame, ctx, "rmse") == {"Floor_one": 1.0}
    assert EnsembleSelector(seed_floor=True)._apply_seed_floor(weights, experts, frame, ctx, "rmse") == weights       # only the seed island


@pytest.mark.slow
def test_a_fit_adds_the_floor_expert_and_the_column_and_predicts_forecast_rows(tmp_path):
    rng = np.random.default_rng(3)
    parts = []
    for g, level in enumerate((20.0, 60.0)):
        n = 110
        ds = pd.date_range("2023-01-01", periods=n)
        parts.append(pd.DataFrame({"ds": ds, "g": g, "x1": rng.normal(size=n),
                                   "y": level + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.3, size=n)}))
    panel = pd.concat(parts, ignore_index=True)
    past, future = panel[panel["ds"] < "2023-04-10"], panel[panel["ds"] >= "2023-04-10"]
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[11], n_experts=1, pop_size=4, use_opera=True, seed=2, pooled_floor=True, export_dir=str(tmp_path))
    model.fit(past, date_col="ds", group_col="g", expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": False, "grid": False})
    assert FLOOR_COLUMN in model.ctx.df_all_aug.columns and FLOOR_COLUMN not in model.ctx.formula_config["features"]
    floors = {e["name"] for e in model.trained_experts if e.get("island") == FLOOR_ISLAND}
    assert floors and floors <= set(model.apex_members_)
    out = model.predict(future.assign(y=np.nan), date_col="ds")
    assert np.isfinite(out[model.final_model_column_]).all()


def test_the_floor_needs_a_date_column_and_a_natural_season(tmp_path):
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=90, freq="3D"), "x1": np.arange(90.0), "y": np.arange(90.0)})
    with pytest.raises(ValueError, match="natural season"):
        AutoTAM("y ~ AutoPipe(x1)", pooled_floor=True, export_dir=str(tmp_path)).fit(frame, date_col="ds")
    with pytest.raises(ValueError, match="date column"):
        AutoTAM("y ~ AutoPipe(x1)", pooled_floor=True, export_dir=str(tmp_path)).fit(frame.drop(columns="ds"))
