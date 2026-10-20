# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Spectral profile of a target: which cycles are present, and how many harmonics each needs.

The profile is computed once on the training rows (no forecast row, no leakage) and is a prior for the search, not a constraint: it names
the cycles of the day, the week and the year found in the periodogram, the order of each, the interactions suggested by modulation
sidebands (a daily profile that changes with the weekday or the season), the periods that are not calendar periods, and the seasonal
autocorrelation at one day, one week and one year.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

#: <spectral_profile>
PEAK_RATIO = 40.0              # a peak is at least this many times the running median of the periodogram (white noise: well under 1 % of series)
MIN_BACKGROUND_BINS = 15       # the running median covers at least this many bins, wider than the main lobe of the window
MIN_PERIOD_STEPS = 2.5
HISTORY_PER_PERIOD = 1.9       # a period longer than history / 1.9 is not resolved
BIN_TOLERANCE = 1.5            # a peak is the harmonic n of a base period when within this many frequency bins of n / base
ORDER_CAPS = {"day": 24, "week": 6, "year": 12}
MIN_ROWS = 24


def _fill_and_detrend(values: np.ndarray) -> Optional[np.ndarray]:
    """Linear interpolation of the gaps, then the linear trend removed; ``None`` when too few values remain."""
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < MIN_ROWS:
        return None
    index = np.arange(len(y))
    y = np.interp(index, index[ok], y[ok])
    slope, intercept = np.polyfit(index, y, 1)
    return y - (slope * index + intercept)


def _seasonal_autocorrelation(values: np.ndarray, lag: int, difference: bool) -> Optional[float]:
    """Correlation of the series with itself ``lag`` steps earlier (on first differences for weekly and coarser data)."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if difference:
        y = np.diff(y)
    if lag < 1 or len(y) < 2 * lag + 2:
        return None
    a, b = y[lag:], y[:-lag]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def profile_series(values: np.ndarray, steps_per_day: float) -> Dict[str, Any]:
    """Cycles of one series.

    Args:
        values: The training values of the series, in time order (gaps as NaN).
        steps_per_day: Number of steps in one day (24 hourly, 1 daily, 1/7 weekly).

    Returns:
        ``{'orders': {'day'|'week'|'year': largest harmonic seen}, 'interactions': {...}, 'other_periods': [steps], 'acf': {...}, 'n': rows}``;
        ``{'n': rows}`` alone when the series is too short or constant.
    """
    y = _fill_and_detrend(values)
    n = len(values)
    if y is None or np.allclose(y, 0):
        return {"n": n, "orders": {}, "interactions": {}, "other_periods": [], "acf": {}}
    bases = {"day": steps_per_day, "week": 7.0 * steps_per_day, "year": 365.25 * steps_per_day}
    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0)
    size = max(MIN_BACKGROUND_BINS, (len(power) // 40) | 1)
    background = median_filter(power, size=size, mode="nearest")
    bin_width = 1.0 / len(y)
    orders: Dict[str, int] = {}
    other: List[float] = []
    sidebands = {"weekday": False, "season": False}
    year_resolved = bases["year"] <= len(y) / HISTORY_PER_PERIOD
    day_usable = bases["day"] >= MIN_PERIOD_STEPS
    peaks = []
    for k in range(2, len(power) - 1):
        if not (power[k] > power[k - 1] and power[k] >= power[k + 1]) or power[k] < PEAK_RATIO * max(background[k], 1e-12):
            continue
        period = 1.0 / freqs[k]
        if MIN_PERIOD_STEPS <= period <= len(y) / HISTORY_PER_PERIOD:
            peaks.append(freqs[k])
    day_harmonics = set()
    rest = []
    for f in peaks:
        harmonic = round(f * bases["day"])
        if day_usable and harmonic >= 1 and abs(f - harmonic / bases["day"]) <= BIN_TOLERANCE * bin_width:
            orders["day"] = max(orders.get("day", 0), min(harmonic, ORDER_CAPS["day"]))
            day_harmonics.add(harmonic)
        else:
            rest.append(f)
    for f in rest:
        near = [(abs(f - h / bases["day"]), h) for h in day_harmonics]
        offset = min(near)[0] if near else None
        if offset is not None and offset <= 3.0 / bases["week"] + BIN_TOLERANCE * bin_width:
            sidebands["weekday"] = True
            continue
        if offset is not None and year_resolved and offset <= 3.0 / bases["year"] + BIN_TOLERANCE * bin_width:
            sidebands["season"] = True
            continue
        week_n = round(f * bases["week"])
        if bases["week"] >= MIN_PERIOD_STEPS and week_n >= 1 and abs(f - week_n / bases["week"]) <= BIN_TOLERANCE * bin_width:
            orders["week"] = max(orders.get("week", 0), min(week_n, ORDER_CAPS["week"]))
            continue
        year_n = round(f * bases["year"])
        if year_resolved and 1 <= year_n <= 2 * ORDER_CAPS["year"] and abs(f - year_n / bases["year"]) <= BIN_TOLERANCE * bin_width:
            orders["year"] = max(orders.get("year", 0), min(year_n, ORDER_CAPS["year"]))
            continue
        other.append(round(1.0 / f, 3))
    difference = steps_per_day <= 1.0 / 7.0
    acf = {}
    for kind, base in bases.items():
        lag = int(round(base))
        value = _seasonal_autocorrelation(values, lag, difference) if base >= MIN_PERIOD_STEPS else None
        acf[kind] = None if value is None else round(value, 4)
    return {
        "n": n, "orders": orders,
        "interactions": {"hour_x_weekday": sidebands["weekday"], "hour_x_season": sidebands["season"]},
        "other_periods": sorted(other)[:5], "acf": acf,
    }


def steps_per_day(dates: pd.Series) -> Optional[float]:
    """Steps in one day from the median step of a date column (``None`` when it cannot be told)."""
    ordered = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()
    delta = ordered.diff().dropna().median()
    if pd.isna(delta) or delta.total_seconds() <= 0:
        return None
    return 86400.0 / delta.total_seconds()


def spectral_profile(df: pd.DataFrame, target: str, date_col: str, group_col: Optional[str] = None, max_groups: int = 20) -> Dict[str, Any]:
    """Cycles of a target on the training rows, over a sample of its groups.

    Args:
        df: Training rows only (never the forecast rows).
        target: Name of the target column.
        date_col: Name of the date column.
        group_col: Name of the group column, if the data are grouped.
        max_groups: At most this many groups are profiled (evenly spaced in sorted order, so the choice is reproducible).

    Returns:
        ``{'steps_per_day', 'cycles': {kind: {'share', 'order', 'acf'}}, 'interactions': {...}, 'other_periods': [...], 'groups': n}``.
        A cycle is present when it is found in at least half of the profiled groups; its order is the largest harmonic seen in at least half
        of the groups that show it.
    """
    spd = steps_per_day(df[date_col])
    empty = {"steps_per_day": spd, "cycles": {}, "interactions": {"hour_x_weekday": False, "hour_x_season": False}, "other_periods": [], "groups": 0}
    if spd is None:
        return empty
    if group_col and group_col in df.columns:
        keys = sorted(df[group_col].dropna().unique(), key=str)
        if len(keys) > max_groups:
            keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, max_groups).round()]
        series = [g.sort_values(date_col)[target].to_numpy() for _, g in df[df[group_col].isin(keys)].groupby(group_col)]
    else:
        series = [df.sort_values(date_col)[target].to_numpy()]
    profiles = [p for p in (profile_series(v, spd) for v in series) if p["orders"] or p["acf"]]
    if not profiles:
        return empty
    half = math.ceil(len(profiles) / 2)
    cycles: Dict[str, Dict[str, Any]] = {}
    for kind in ORDER_CAPS:
        showing = [p["orders"][kind] for p in profiles if kind in p["orders"]]
        if len(showing) >= half:
            order = max(o for o in set(showing) if sum(s >= o for s in showing) >= math.ceil(len(showing) / 2))
            acfs = [p["acf"].get(kind) for p in profiles if p["acf"].get(kind) is not None]
            cycles[kind] = {"share": round(len(showing) / len(profiles), 3), "order": int(order), "acf": round(float(np.mean(acfs)), 4) if acfs else None}
    flags = {key: sum(p["interactions"].get(key, False) for p in profiles) >= half for key in ("hour_x_weekday", "hour_x_season")}
    other = sorted({period for p in profiles for period in p["other_periods"]})[:5]
    return {"steps_per_day": spd, "cycles": cycles, "interactions": flags, "other_periods": other, "groups": len(profiles)}
#: </spectral_profile>
