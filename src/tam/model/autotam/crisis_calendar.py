# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
A dated calendar of crises, as features.

A lockdown or an energy crisis distorts what a model learns from the history (read as a level or a seasonal change) and, when the
forecast window falls inside one, moves the level the model has to forecast. The dates of such events are facts known in advance, like
holidays. ``crisis_calendar`` takes named presets and the caller's own events (a strike, a plant outage, a tariff change) and writes,
for each event that overlaps the data, an indicator ``crisis_<name>`` and the time since its start ``crisis_<name>_age``.
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

#: <crisis_calendar>
# Worldwide windows, approximate: the day the event is usually dated from, and the end of its acute phase. A country whose lockdown dates differ
# should pass its own events.
PRESETS: Dict[str, Dict[str, str]] = {
    "covid19": {"start": "2020-03-11", "end": "2021-12-31"},         # WHO declaration of the pandemic, to the end of 2021
    "recession_2008": {"start": "2007-12-01", "end": "2009-06-30"},  # NBER dating of the US recession
    "energy_2022": {"start": "2021-10-01", "end": "2023-03-31"},     # the European wholesale price surge, to the end of winter 2022-23
}
AGE_CAP_YEARS = 2.0


def resolve_events(spec: Union[bool, str, Iterable[Any], None]) -> List[Dict[str, Any]]:
    """The events of a ``crisis_calendar`` argument.

    Args:
        spec: ``None`` / ``False`` (none), ``True`` or ``"all"`` (every preset), a preset name, or a list of preset names and of dicts
            ``{"name": ..., "start": date, "end": date}`` (``end`` may be omitted: the event is still on).

    Raises:
        ValueError: Unknown preset, an event without name or start, or an end before its start.
    """
    if spec is None or spec is False:
        return []
    if spec is True or spec == "all":
        items: List[Any] = list(PRESETS)
    elif isinstance(spec, str):
        items = [spec]
    else:
        items = list(spec)
    events = []
    for item in items:
        if isinstance(item, str):
            if item not in PRESETS:
                raise ValueError(f"Unknown crisis preset {item!r}; the presets are {sorted(PRESETS)}, or pass events as dicts.")
            event = {"name": item, **PRESETS[item]}
        elif isinstance(item, dict) and item.get("name") and item.get("start"):
            event = dict(item)
        else:
            raise ValueError(f"A crisis event is a preset name or a dict with 'name' and 'start': got {item!r}.")
        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"]) if event.get("end") else pd.Timestamp.max.normalize()
        if end < start:
            raise ValueError(f"The crisis {event['name']!r} ends before it starts.")
        events.append({"name": str(event["name"]), "start": start, "end": end})
    return events


def active_events(events: List[Dict[str, Any]], dates: pd.Series) -> List[Dict[str, Any]]:
    """The events that overlap the dates (an event outside the data is a constant column and is left out)."""
    stamps = pd.to_datetime(pd.Series(dates)).dropna()
    if stamps.empty:
        return []
    low, high = stamps.min(), stamps.max()
    return [e for e in events if e["start"] <= high and e["end"] >= low]


def crisis_columns(dates: pd.Series, events: List[Dict[str, Any]]) -> pd.DataFrame:
    """``crisis_<name>`` (1 inside the event) and ``crisis_<name>_age`` (years since the start, 0 outside, capped) per event."""
    stamps = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    out: Dict[str, np.ndarray] = {}
    for event in events:
        inside = ((stamps >= event["start"]) & (stamps <= event["end"])).to_numpy()
        years = (stamps - event["start"]).dt.total_seconds().to_numpy() / (365.25 * 86400.0)
        out[f"crisis_{event['name']}"] = inside.astype(float)
        out[f"crisis_{event['name']}_age"] = np.where(inside, np.clip(years, 0.0, AGE_CAP_YEARS), 0.0)
    return pd.DataFrame(out)


def add_crisis(df: pd.DataFrame, date_col: str, events: List[Dict[str, Any]], names: Optional[List[str]] = None) -> List[str]:
    """Adds the crisis columns of ``events`` to ``df`` in place (the ones in ``names`` when given); returns the names added."""
    new = crisis_columns(df[date_col], events)
    added = []
    for name in new.columns:
        if names is not None and name not in names:
            continue
        df[name] = new[name].to_numpy()
        added.append(name)
    return added
#: </crisis_calendar>
