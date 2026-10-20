# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Covariates ranked by their Spearman correlation with the target before the search draws among them (opt-in ``rank_covariates``, ``max_covariates``).
"""

import random

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.covariate_ranking import rank_covariates
from tam.model.autotam.population_nodes import _draw_features

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _wind(n=600, seed=0):
    """A wind farm: the power follows the wind speed; two temperatures say almost nothing."""
    rng = np.random.default_rng(seed)
    wspd = rng.weibull(2.0, n) * 8
    etmp, itmp = rng.normal(15, 8, n), rng.normal(20, 8, n)
    pab1 = rng.uniform(0, 30, n)
    power = 0.5 * wspd ** 2 - 0.02 * pab1 + rng.normal(scale=2.0, size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="h"), "Etmp": etmp, "Itmp": itmp, "Pab1": pab1, "Wspd": wspd, "power": power})


def test_the_wind_speed_ranks_above_the_temperatures_whatever_the_order_of_the_columns():
    frame = _wind()
    table = rank_covariates(frame, "power", ["Etmp", "Itmp", "Pab1", "Wspd"])
    assert table["feature"].iloc[0] == "Wspd" and table["score"].iloc[0] > 0.8
    assert table.set_index("feature").loc[["Etmp", "Itmp"], "score"].max() < 0.15
    shuffled = rank_covariates(frame, "power", ["Wspd", "Itmp", "Etmp", "Pab1"])
    assert list(shuffled["feature"]) == list(table["feature"])


def test_the_irradiance_ranks_first_on_a_solar_series_and_a_twin_of_it_is_redundant():
    rng = np.random.default_rng(1)
    n = 500
    ghi = np.clip(rng.normal(300, 200, n), 0, None)
    frame = pd.DataFrame({"day_length": rng.normal(12, 2, n), "humidity": rng.uniform(20, 90, n), "ghi": ghi, "ghi_copy": ghi * 1.01 + 0.5})
    frame["power"] = 0.9 * ghi + rng.normal(scale=20, size=n)
    table = rank_covariates(frame, "power", ["day_length", "humidity", "ghi", "ghi_copy"]).set_index("feature")
    assert table["score"].idxmax() in ("ghi", "ghi_copy")
    twins = [f for f in ("ghi", "ghi_copy") if pd.notna(table.loc[f, "redundant_with"])]
    assert len(twins) == 1 and table.loc[twins[0], "redundant_with"] in ("ghi", "ghi_copy")


def test_the_target_is_standardised_inside_each_group():
    rng = np.random.default_rng(2)
    rows = []
    for group, level in ((0, 10.0), (1, 5000.0)):
        x = rng.normal(size=200)
        rows.append(pd.DataFrame({"g": group, "x": x, "noise": rng.normal(size=200), "y": level + (3 * level / 10) * x + rng.normal(scale=level / 20, size=200)}))
    table = rank_covariates(pd.concat(rows, ignore_index=True), "y", ["x", "noise"], "g")
    assert table["feature"].iloc[0] == "x" and table["score"].iloc[0] > 0.8


def test_a_constant_or_text_column_scores_zero():
    frame = pd.DataFrame({"a": np.arange(100.0), "c": 1.0, "t": ["u"] * 100, "y": np.arange(100.0)})
    table = rank_covariates(frame, "y", ["a", "c", "t"]).set_index("feature")
    assert table.loc["c", "score"] == 0.0 and table.loc["t", "score"] == 0.0 and table.loc["a", "score"] > 0.99


def test_the_draw_is_uniform_without_importances_and_follows_them_with():
    features = ["a", "b", "c", "d"]
    random.seed(0)
    plain = [_draw_features(features, {}, 1) for _ in range(2000)]
    assert max(plain.count([f]) for f in features) < 0.4 * 2000 and min(plain.count([f]) for f in features) > 0.15 * 2000
    space = {"a": {"importance": 0.9}, "b": {"importance": 0.0}, "c": {"importance": 0.0}, "d": {"importance": 0.0}}
    random.seed(0)
    weighted = [_draw_features(features, space, 1)[0] for _ in range(2000) if True]
    assert weighted.count("a") > 0.5 * len(weighted) and weighted.count("b") > 0.03 * len(weighted)      # the floor keeps the others in play


def _model(directory, **options):
    return AutoTAM("power ~ AutoPipe(Etmp, Itmp, Pab1, Wspd)", n_experts=2, pop_size=8, use_opera=False, seed=3, export_dir=str(directory), **options)


def test_the_ranking_is_recorded_and_the_extra_covariates_are_not_offered(tmp_path):
    frame = _wind()
    model = _model(tmp_path, max_covariates=2).fit(frame.iloc[:500], date_col="ds", expansions=EXPANSIONS)
    table = model.covariate_ranking_
    assert table["feature"].iloc[0] == "Wspd" and bool(table["offered"].iloc[0])
    offered = [f for f in model.ctx.search_space if f not in ("mandatory_terms", "mandatory_variables")]      # the engineered features count too
    assert len(offered) == 2 and "Wspd" in offered
    assert int(table["offered"].sum()) == 2
    formulas = " ".join(f for group in model.architectures_.values() for f in group)
    assert "Wspd" in formulas


def test_the_importance_reaches_the_search_space(tmp_path):
    model = _model(tmp_path, rank_covariates=True).fit(_wind().iloc[:500], date_col="ds", expansions=EXPANSIONS)
    assert model.ctx.search_space["Wspd"]["importance"] > 0.8 and model.ctx.search_space["Etmp"]["importance"] < 0.2


def test_the_default_does_not_rank(tmp_path):
    model = _model(tmp_path).fit(_wind().iloc[:500], date_col="ds", expansions=EXPANSIONS)
    assert model.covariate_ranking_.empty and "importance" not in model.ctx.search_space["Wspd"]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_maximum_that_is_not_a_positive_whole_number_is_refused(bad, tmp_path):
    with pytest.raises(ValueError, match="max_covariates"):
        _model(tmp_path, max_covariates=bad)
