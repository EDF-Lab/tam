# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Calendar features derived from the date column, and the seed formulas the spectral profile suggests.

Calendar positions are known in advance, so they are legitimate known covariates (no leakage). They are positions on a cycle, never linear
terms: ``toy`` (time of year) and ``tod`` (time of day) are fractions in [0, 1) used with cyclic Fourier terms, ``dow`` (day of week) is
an integer 0 to 6. Country holidays come from the optional ``holidays`` package (``pip install tam-ml[calendar]``) and give a holiday flag,
the days to the next and since the last holiday (capped), and a bridge-day flag.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: <calendar_features>
HOLIDAY_COLUMNS = ("holiday", "days_to_holiday", "days_since_holiday", "bridge_day")
HOLIDAY_DISTANCE_CAP = 14


def calendar_names(steps_per_day: Optional[float]) -> List[str]:
    """Calendar columns that make sense for a sampling step: ``tod`` below a day, ``dow`` from a day down to a day, ``toy`` always."""
    names = ["toy"]
    if steps_per_day is not None and steps_per_day > 1.0:
        names.insert(0, "tod")
    if steps_per_day is not None and steps_per_day >= 1.0:
        names.append("dow")
    return names


def calendar_columns(dates: pd.Series, names: List[str]) -> pd.DataFrame:
    """The calendar positions of a date column (``toy`` and ``tod`` in [0, 1), ``dow`` from 0 to 6)."""
    d = pd.to_datetime(dates)
    seconds = (d.dt.hour * 3600 + d.dt.minute * 60 + d.dt.second).to_numpy(dtype=float)
    out = {}
    if "tod" in names:
        out["tod"] = seconds / 86400.0
    if "toy" in names:
        days_in_year = np.where(d.dt.is_leap_year.to_numpy(), 366.0, 365.0)
        out["toy"] = (d.dt.dayofyear.to_numpy(dtype=float) - 1.0 + seconds / 86400.0) / days_in_year
    if "dow" in names:
        out["dow"] = d.dt.dayofweek.to_numpy(dtype=float)
    return pd.DataFrame(out, index=dates.index)


def _holiday_calendar(country: str, subdivision: Optional[str], years: List[int]):
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("country= needs the 'holidays' package: pip install tam-ml[calendar] (or pip install holidays).") from exc
    try:
        return holidays.country_holidays(country, subdiv=subdivision, years=years)
    except (NotImplementedError, KeyError) as exc:
        raise ValueError(f"The holidays package has no calendar for country={country!r}, subdivision={subdivision!r}.") from exc


def holiday_columns(dates: pd.Series, country: str, subdivision: Optional[str] = None) -> pd.DataFrame:
    """Holiday flag, days to the next and since the last holiday (capped at 14) and bridge-day flag of a date column.

    A bridge day is a working Monday or Friday next to a holiday on Tuesday or Thursday (the day between the holiday and the weekend).
    """
    d = pd.to_datetime(dates).dt.normalize()
    years = list(range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2))
    calendar = _holiday_calendar(country, subdivision, years)
    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar.keys()))))
    flag = d.isin(days).to_numpy()
    values = d.to_numpy()
    nxt = np.full(len(d), HOLIDAY_DISTANCE_CAP, dtype=float)
    prev = np.full(len(d), HOLIDAY_DISTANCE_CAP, dtype=float)
    position = np.searchsorted(days.to_numpy(), values, side="left")
    for i, p in enumerate(position):
        if p < len(days):
            nxt[i] = min(HOLIDAY_DISTANCE_CAP, (days[p] - pd.Timestamp(values[i])).days)
        q = np.searchsorted(days.to_numpy(), values[i], side="right") - 1
        if q >= 0:
            prev[i] = min(HOLIDAY_DISTANCE_CAP, (pd.Timestamp(values[i]) - days[q]).days)
    dow = d.dt.dayofweek.to_numpy()
    next_day = (d + pd.Timedelta(days=1)).isin(days).to_numpy()
    previous_day = (d - pd.Timedelta(days=1)).isin(days).to_numpy()
    bridge = ~flag & (((dow == 0) & next_day) | ((dow == 4) & previous_day))
    return pd.DataFrame({"holiday": flag.astype(float), "days_to_holiday": nxt, "days_since_holiday": prev, "bridge_day": bridge.astype(float)},
                        index=dates.index)


EQUIVALENT_CORRELATION = 0.999


def has_equivalent(frame: pd.DataFrame, values: np.ndarray, name: str) -> Optional[str]:
    """The name of a numeric column of ``frame`` that carries the same information as ``values`` (same up to a linear change), else ``None``.

    A caller who already gives the hour, the day of week or a holiday flag does not need ours: the two would be collinear terms of the search.
    """
    v = np.asarray(values, dtype=float)
    if np.nanstd(v) == 0:
        return None
    for column in frame.columns:
        if column == name or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        other = frame[column].to_numpy(dtype=float)
        if np.nanstd(other) == 0 or np.isnan(other).all():
            continue
        both = np.isfinite(v) & np.isfinite(other)
        if both.sum() > 2 and abs(np.corrcoef(v[both], other[both])[0, 1]) >= EQUIVALENT_CORRELATION:
            return str(column)
    return None


def add_calendar(df: pd.DataFrame, date_col: str, spec: Dict[str, Any], skip_equivalent: bool = False) -> List[str]:
    """Adds the calendar and holiday columns of ``spec`` that the frame does not already have (in place); returns the names added.

    With ``skip_equivalent`` (the fit) a column is not added when the frame already has one that carries the same information under another name
    (an hour column for ``tod``, a day-of-week column for ``dow``, a holiday flag for ``holiday``); the names kept are then in ``spec['keep']``,
    which the later frames (the forecast rows) follow, so the columns of the fit and of the prediction are the same.
    """
    added = []
    wanted = list(spec.get("names", []))
    new = calendar_columns(df[date_col], wanted) if wanted else pd.DataFrame(index=df.index)
    if spec.get("country"):
        new = pd.concat([new, holiday_columns(df[date_col], spec["country"], spec.get("subdivision"))], axis=1)
    keep = spec.get("keep")
    for name in new.columns:
        if name in df.columns or (keep is not None and name not in keep):
            continue
        if skip_equivalent and has_equivalent(df, new[name].to_numpy(), name):
            logger.info("Calendar column '%s' not added: the frame already has an equivalent column.", name)
            continue
        df[name] = new[name].to_numpy()
        added.append(name)
    if skip_equivalent:
        spec["keep"] = list(added)
    return added


def calendar_seed_formulas(target: str, profile: Dict[str, Any], columns: List[str], holidays_on: bool = False) -> List[str]:
    """Seed formulas for the cycles the spectral profile found (empty when none is present).

    One formula holds a cyclic Fourier term for every cycle present (``tod`` for the day, ``dow`` for the week, ``toy`` for the year), a
    second adds the tensor product the profile flagged (time of day by day of week). With holidays, the holiday and bridge flags join them.
    """
    cycles = profile.get("cycles", {})
    parts = []
    if "day" in cycles and "tod" in columns:
        parts.append(f"f(tod, m={cycles['day']['order']}, cyclic=True, period=(0, 1))")
    if "week" in cycles and "dow" in columns:
        parts.append(f"f(dow, m={min(cycles['week']['order'], 3)}, cyclic=True, period='auto')")
    if "year" in cycles and "toy" in columns:
        parts.append(f"f(toy, m={cycles['year']['order']}, cyclic=True, period=(0, 1))")
    if not parts:
        return []
    extra = ["l(holiday)", "l(bridge_day)"] if holidays_on and "holiday" in columns else []
    formulas = [f"{target} ~ " + " + ".join(parts + extra)]
    if profile.get("interactions", {}).get("hour_x_weekday") and "day" in cycles and "tod" in columns and "dow" in columns:
        formulas.append(f"{target} ~ " + " + ".join(parts + extra + [f"te(f(tod, m={cycles['day']['order']}, cyclic=True, period=(0, 1)), c(dow, topo='nominal', n_cat=7))"]))
    return formulas
#: </calendar_features>
