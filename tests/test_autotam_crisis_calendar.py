# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The dated crisis calendar (opt-in): events known in advance, as an indicator and an age, offered to the search like holidays.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.crisis_calendar import PRESETS, active_events, add_crisis, crisis_columns, resolve_events

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def test_a_preset_a_list_and_all_resolve_to_dated_events():
    assert [e["name"] for e in resolve_events("covid19")] == ["covid19"]
    assert {e["name"] for e in resolve_events(True)} == set(PRESETS) == {e["name"] for e in resolve_events("all")}
    own = resolve_events(["energy_2022", {"name": "strike", "start": "2023-03-07", "end": "2023-04-20"}])
    assert [e["name"] for e in own] == ["energy_2022", "strike"] and own[1]["start"] == pd.Timestamp("2023-03-07")
    assert resolve_events(None) == [] and resolve_events(False) == []


@pytest.mark.parametrize("bad", ["lockdown", [{"name": "x"}], [{"name": "x", "start": "2023-02-01", "end": "2023-01-01"}], [3]])
def test_a_bad_event_is_refused(bad):
    with pytest.raises(ValueError):
        resolve_events(bad)


def test_the_columns_are_one_inside_the_event_and_the_age_grows_from_its_start():
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="D"))
    events = resolve_events([{"name": "e", "start": "2023-01-03", "end": "2023-01-06"}])
    out = crisis_columns(dates, events)
    assert out["crisis_e"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    age = out["crisis_e_age"].to_numpy()
    assert age[:2].sum() == 0 and age[6:].sum() == 0 and age[2] == 0 and np.all(np.diff(age[2:6]) > 0)


def test_an_event_outside_the_data_is_left_out_and_an_open_event_is_still_on():
    dates = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
    events = resolve_events([{"name": "old", "start": "2010-01-01", "end": "2010-02-01"}, {"name": "open", "start": "2023-01-20"}])
    assert [e["name"] for e in active_events(events, dates)] == ["open"]
    assert crisis_columns(dates, active_events(events, dates))["crisis_open"].iloc[-1] == 1


def test_add_crisis_writes_only_the_columns_asked_for():
    frame = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=5)})
    events = resolve_events([{"name": "e", "start": "2023-01-02"}])
    assert add_crisis(frame, "ds", events, names=["crisis_e"]) == ["crisis_e"] and list(frame.columns) == ["ds", "crisis_e"]


def _series(n=520, seed=2):
    """A level of 50 with a weekly cycle, 15 lower inside the event (days 250 to 340)."""
    rng = np.random.default_rng(seed)
    day = np.arange(n)
    drop = np.where((day >= 250) & (day <= 340), -15.0, 0.0)
    x1 = rng.normal(size=n)
    y = 50 + 3 * x1 + 4 * np.sin(2 * np.pi * day / 7) + drop + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n), "x1": x1, "y": y})


def _fit(tmp_path, label, **kwargs):
    frame = _series()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=6, use_opera=False, seed=4, export_dir=str(tmp_path / label), **kwargs)
    model.fit(frame.iloc[:420], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_the_event_explains_the_drop_and_the_level_after_it_is_forecast_better(tmp_path):
    event = [{"name": "outage", "start": "2022-09-08", "end": "2022-12-07"}]   # days 250 to 340 of the series
    plain, frame = _fit(tmp_path, "plain")
    flagged, _ = _fit(tmp_path, "flagged", crisis_calendar=event)
    assert "crisis_outage" in flagged.ctx.df_all_aug.columns and "crisis_outage" not in plain.ctx.df_all_aug.columns
    test = frame.iloc[420:]
    errors = {}
    for label, model in (("plain", plain), ("flagged", flagged)):
        forecast = model.predict(test, date_col="ds")[model.trained_experts[0]["name"]].to_numpy()
        errors[label] = float(np.sqrt(np.mean((forecast - test["y"].to_numpy()) ** 2)))
    assert errors["flagged"] < 0.8 * errors["plain"], errors


def test_the_columns_of_the_fit_are_the_columns_of_the_forecast_rows(tmp_path):
    event = [{"name": "outage", "start": "2022-09-08", "end": "2022-12-07"}]
    model, frame = _fit(tmp_path, "cols", crisis_calendar=event)
    future = frame.iloc[420:430].copy()
    future["y"] = np.nan
    out = model.predict(future, date_col="ds")
    assert len(out) == 10 and np.isfinite(out[model.trained_experts[0]["name"]].to_numpy()).all()


def test_no_event_in_the_data_changes_nothing(tmp_path):
    model, _ = _fit(tmp_path, "none", crisis_calendar=["covid19"])
    assert model.ctx.crisis == {} and not any(c.startswith("crisis_") for c in model.ctx.df_all_aug.columns)
