# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Adaptive configurations of the online variants: the forecast and the effects with the calendar flags switched on (holidays, crises), and the cost
of a fit reported by phase and per series.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.effect_groups import calendar_flags

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": True, "grid": False}


def test_the_calendar_flags_are_the_holiday_columns_and_the_crisis_indicators():
    columns = ["y", "x1", "holiday", "bridge_day", "crisis_strike", "crisis_strike_age", "toy"]
    assert calendar_flags(columns) == ["holiday", "bridge_day", "crisis_strike"]
    assert calendar_flags(["y", "x1"]) == []


def _daily(n=240, seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    x1 = rng.normal(size=n)
    strike = ((dates >= "2023-04-01") & (dates <= "2023-04-30")).astype(float)
    y = 30 + 2 * x1 - 6 * strike + 2 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": dates, "x1": x1, "y": y})


def _model(directory, **options):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=True, seed=5, dynamic_top=1, export_dir=str(directory), **options)


def test_the_flag_configuration_is_built_with_online_variants_and_a_calendar_event(tmp_path):
    model = _model(tmp_path, online_variants=True, crisis_calendar=[{"name": "strike", "start": "2023-04-01", "end": "2023-04-30"}])
    model.fit(_daily(), date_col="ds", expansions=EXPANSIONS)
    flagged = [row for row in model.chronological_test_log if "ECM_Flags" in str(row.get("Model_Name", ""))]
    assert flagged and all("crisis_strike" in row["Formula"] and "c(crisis_strike" in row["Formula"] for row in flagged)


def test_without_online_variants_there_is_no_flag_configuration(tmp_path):
    model = _model(tmp_path, crisis_calendar=[{"name": "strike", "start": "2023-04-01", "end": "2023-04-30"}])
    model.fit(_daily(), date_col="ds", expansions=EXPANSIONS)
    assert not [row for row in model.chronological_test_log if "ECM_Flags" in str(row.get("Model_Name", ""))]


def test_the_cost_of_a_fit_is_reported_by_phase_and_per_series(tmp_path):
    model = _model(tmp_path).fit(_daily(), date_col="ds", expansions=EXPANSIONS)
    timings = model.timings_
    assert set(timings) >= {"search", "expansion", "selection", "total", "series", "seconds_per_series"}
    assert timings["series"] == 1 and timings["total"] > 0
    assert abs(timings["search"] + timings["expansion"] + timings["selection"] - timings["total"]) < 1e-6
    recorded = json.loads(next(Path(tmp_path).glob("AutoTAM_run_info_*.json")).read_text(encoding="utf-8"))
    assert recorded["timings"]["total"] == timings["total"]


def test_the_cost_per_series_divides_by_the_groups(tmp_path):
    rng = np.random.default_rng(4)
    frames = []
    for site in ("a", "b", "c"):
        frame = _daily(seed=int(rng.integers(100)))
        frame["site"] = site
        frames.append(frame)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, export_dir=str(tmp_path))
    model.fit(pd.concat(frames, ignore_index=True), date_col="ds", group_col="site", expansions={**EXPANSIONS, "adaptive": False})
    assert model.timings_["series"] == 3
    assert model.timings_["seconds_per_series"] == model.timings_["total"] / 3


def test_a_group_search_reports_the_time_of_every_version(tmp_path):
    rng = np.random.default_rng(3)
    n = 24 * 90
    hour = np.arange(n) % 24
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="h"), "x1": x1, "y": 50 + (1 + hour / 6) * x1 + rng.normal(scale=0.3, size=n)})
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, seed=5, group_search=True, lags=[24], export_dir=str(tmp_path))
    model.fit(frame, date_col="ds", expansions={**EXPANSIONS, "adaptive": False})
    assert model.timings_["group_search_total"] >= model.timings_["total"] > 0
