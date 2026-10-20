# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Seasonal anchor (opt-in): ``AutoTAM(..., seasonal_anchor=True)``.

On many series the strongest simple reference is the last season repeated. A model that has more information than that reference should never be
clearly worse than it, so the anchor is put in the pool as an expert of its own: a regression of the target on its value one season earlier
(``y ~ l(anchor_lag_<season>)``), expanded like any champion (prior, Kalman, adaptive) under the island name ``Anchor``. The season comes from the
sampling step of the data (24 hourly, 7 daily, 52 weekly, 12 monthly, 4 quarterly) or from ``season=``.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import anchor
from tam.model.autotam.auto_tam import AutoTAM


def test_the_constructor_has_defaulted_anchor_arguments():
    parameters = inspect.signature(AutoTAM.__init__).parameters
    assert parameters["seasonal_anchor"].default is False
    assert parameters["season"].default is None


@pytest.mark.parametrize("freq, expected", [("h", 24), ("H", 24), ("60min", 24), ("30min", 48), ("15T", 96), ("D", 7), ("1D", 7), ("W", 52),
                                            ("W-SUN", 52), ("MS", 12), ("ME", 12), ("QS", 4), ("YS", None), ("3D", None)])
def test_the_natural_season_comes_from_the_sampling_step(freq, expected):
    assert anchor.natural_season(freq) == expected


def test_an_explicit_season_wins_and_must_be_a_positive_integer():
    assert anchor.resolve_season(12, "D", []) == 12
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            anchor.resolve_season(bad, "D", [])


def test_a_step_without_a_natural_season_asks_for_one():
    with pytest.raises(ValueError) as error:
        anchor.resolve_season(None, "3D", [])
    assert "season=" in str(error.value)


def test_a_season_shorter_than_the_horizon_is_rejected():
    """The season is a lag of the target: below the horizon (the smallest target lag) its value would not be known at the forecast origin."""
    with pytest.raises(ValueError) as error:
        anchor.resolve_season(7, "D", [24])
    assert "7" in str(error.value) and "24" in str(error.value) and "horizon" in str(error.value)


def test_the_anchor_formula_and_column_names():
    assert anchor.anchor_column(7) == "anchor_lag_7"
    assert anchor.anchor_formula("y", 7) == "y ~ l(anchor_lag_7)"


def test_the_lag_is_taken_inside_each_group():
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10, "y": list(range(10)) + list(range(100, 110))})
    anchor.add_anchor_lag(df, "y", 3, "g")
    assert df.loc[df["g"] == "a", "anchor_lag_3"].tolist()[3:] == list(range(0, 7))
    assert df.loc[df["g"] == "b", "anchor_lag_3"].tolist()[3:] == list(range(100, 107))
    assert df.loc[df["g"] == "b", "anchor_lag_3"].isna().sum() == 3


def _seasonal(n=210):
    rng = np.random.default_rng(1)
    ds = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"ds": ds, "x1": rng.normal(size=n)})
    df["y"] = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.5 * df["x1"] + rng.normal(scale=0.5, size=n)
    return df


EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": False, "grid": False}


@pytest.fixture(scope="module")
def anchored(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_anchor"))
    try:
        df = _seasonal()
        model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=True, seasonal_anchor=True)
        model.fit(df.iloc[:180].copy(), date_col="ds", expansions=EXPANSIONS)
        yield model, df
    finally:
        os.chdir(previous)


def test_the_context_knows_the_season(anchored):
    model, _ = anchored
    assert model.ctx.anchor_season == 7
    assert "anchor_lag_7" in model.ctx.df_all_aug.columns


def test_the_anchor_is_an_expert_in_the_board_and_in_the_apex_pool(anchored):
    model, _ = anchored
    names = [e["name"] for e in model.trained_experts]
    assert any(n.startswith("Static_Anchor_R1") for n in names)
    assert any("Anchor_R1" in n for n in model.apex_members_)


def test_the_anchor_expert_forecasts_a_seasonal_series_well(anchored):
    model, df = anchored
    test = df.iloc[180:]
    out = model.predict(test, date_col="ds")
    column = next(c for c in out.columns if c.startswith("Static_Anchor_R1"))
    rmse = float(np.sqrt(np.mean((out[column].to_numpy() - test["y"].to_numpy()) ** 2)))
    assert rmse < 2.0   # noise 0.5 plus the part of x1 the anchor does not see


def test_without_the_argument_there_is_no_anchor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False)
    model.fit(_seasonal(90).copy(), date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    assert model.ctx.anchor_season is None
    assert "anchor_lag_7" not in model.ctx.df_all_aug.columns
    assert not [e for e in model.trained_experts if "Anchor" in e["name"]]
