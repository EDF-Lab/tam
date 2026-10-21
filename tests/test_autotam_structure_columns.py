# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Columns that are not ordinary covariates: an exposure offset (``offset_col``), a pair of coordinates (``spatial_pairs``), an identifier with many
levels (``random_effects``).
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.feature_engineer import FeatureEngineer
from tam.model.autotam.population_nodes import _categorical_interaction_partners
from tam.model.autotam.structure_columns import random_effect_columns, spatial_pairs, with_offset

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def test_the_coordinate_pairs_are_found_whatever_the_case():
    assert spatial_pairs(["Lon", "Lat", "depth"]) == [("Lon", "Lat")]
    assert spatial_pairs(["X", "Y", "E", "N"]) == [("X", "Y"), ("E", "N")]
    assert spatial_pairs(["lon", "depth"]) == [] and spatial_pairs(["x", "z"]) == []


def test_the_offset_is_written_once_and_only_when_asked():
    assert with_offset("y ~ s(x)", "log_a") == "y ~ s(x) + offset(log_a)"
    assert with_offset("y ~ s(x) + offset(log_a)", "log_a") == "y ~ s(x) + offset(log_a)"
    assert with_offset("y ~ s(x)", None) == "y ~ s(x)"


def _panel(subjects=50, rows=10, seed=0):
    rng = np.random.default_rng(seed)
    n = subjects * rows
    return pd.DataFrame({"id": np.repeat(np.arange(1, subjects + 1), rows), "t": np.tile(np.arange(rows), subjects).astype(float),
                         "counter": np.arange(n), "age": rng.integers(20, 70, n), "month": rng.integers(1, 13, n),
                         "site": rng.choice([f"s{i}" for i in range(30)], n), "x": rng.normal(size=n), "y": rng.normal(size=n)})


def test_an_identifier_with_many_levels_is_found_and_a_counter_a_month_or_an_age_is_not():
    found = random_effect_columns(_panel(), exclude=["y"])
    assert "id" in found and "site" in found
    assert "counter" not in found and "month" not in found and "age" not in found and "t" not in found and "x" not in found


def test_a_perfectly_balanced_panel_column_without_an_id_word_is_found():
    frame = pd.DataFrame({"unit": np.repeat(np.arange(40), 5), "v": np.random.default_rng(1).normal(size=200)})
    assert random_effect_columns(frame) == ["unit"]


def test_the_engineer_makes_the_ratio_of_the_first_two_columns_unless_they_are_declared():
    frame = pd.DataFrame({"lon": np.linspace(1, 2, 60), "lat": np.linspace(3, 4, 60) ** 2, "y": np.arange(60.0)})
    config, metadata = {"targets": ["y"], "features": ["lon", "lat"]}, {"is_time_series": False}
    assert "lon_div_lat" in FeatureEngineer().engineer_features(frame, config, metadata).columns
    engineer = FeatureEngineer()
    engineer.no_ratio.update({"lon", "lat"})
    assert not any("_div_" in c for c in engineer.engineer_features(frame, config, metadata).columns)


def test_an_identifier_is_a_ridge_intercept_and_the_partner_of_a_factor_smooth():
    frame = _panel()
    frame["y"] = frame["x"] + np.random.default_rng(2).normal(size=len(frame))
    config = {"targets": ["y"], "features": ["id", "t", "x"], "random_effect_columns": ["id"]}
    space = EffectSelector().build_search_space(frame, config, {})
    assert space["id"]["eligible_effects"] == ["c"] and space["id"]["grids"]["c"]["ap"] == [-2.0] and space["id"]["grids"]["c"]["n_cat"] == [51]
    assert "id" in _categorical_interaction_partners(["id", "x"], space)


def _spatial(n=400, seed=5):
    rng = np.random.default_rng(seed)
    lon, lat = rng.uniform(0, 10, n), rng.uniform(40, 50, n)
    effort = rng.uniform(0.5, 3.0, n)
    mean = np.exp(0.3 * np.sin(lon / 2) + 0.2 * np.cos((lat - 40) / 3)) * effort
    return pd.DataFrame({"lon": lon, "lat": lat, "log_effort": np.log(effort), "y": rng.poisson(mean * 3).astype(float)})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(lon, lat)", n_experts=1, pop_size=4, use_opera=False, seed=2, export_dir=str(directory), **options)


def test_the_offset_is_in_every_formula_and_not_among_the_features(tmp_path):
    model = _model(tmp_path, offset_col="log_effort").fit(_spatial(), expansions=EXPANSIONS)
    formulas = [f for group in model.architectures_.values() for f in group]
    assert formulas and all("offset(log_effort)" in f for f in formulas)
    assert "log_effort" not in model.ctx.search_space
    assert not any("log_effort_div" in c or "div_log_effort" in c for c in model.ctx.df_fit.columns)
    out = model.predict(_spatial(60, seed=9))
    assert np.isfinite(out[model.trained_experts[0]["name"]].to_numpy()).all()


def test_the_spatial_pair_gets_no_ratio_and_two_seed_surfaces(tmp_path):
    model = _model(tmp_path, spatial_pairs=True).fit(_spatial(), expansions=EXPANSIONS)
    assert model.spatial_pairs_ == [("lon", "lat")]
    assert not any("lon_div_lat" in c or "lat_div_lon" in c for c in model.ctx.df_fit.columns)
    seeds = [f for f in model.architectures_.get("Seed", [])]
    assert any("kernel='tps'" in f for f in seeds) and any(f.startswith("y ~ te(s(lon") for f in seeds)


def test_the_default_leaves_the_columns_alone(tmp_path):
    model = _model(tmp_path).fit(_spatial(), expansions=EXPANSIONS)
    assert model.spatial_pairs_ == [] and model.random_effect_columns_ == [] and model.ctx.offset_col is None
