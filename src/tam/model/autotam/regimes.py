# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Regimes detected in the training target: level shifts such as a lockdown, a crisis or a change of the process.

The breaks are found on the training rows only, by binary segmentation of the level of the target (seasonal profile removed), and kept as
dates. A row gets the number of the regime its date falls in; a forecast row, dated after the last break, carries the regime in force at the
cutoff, so no feature of a forecast row depends on data after the cutoff. The regime is offered to the search as a category.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: <regimes>
BREAK_THRESHOLD = 7.0       # standardised difference of the means of two segments; white noise stays far below it
MIN_SEGMENT_SHARE = 0.05    # a regime is at least this share of the series ...
MIN_SEGMENT_ROWS = 10       # ... and at least this many rows
MAX_REGIMES = 6
MIN_SEGMENT_SEASONS = 4     # a regime lasts at least this many seasons (a day, a week, a year, by the sampling step)
LAST_REGIME_SEASONS = 8     # the last regime is trusted for the forecast rows once it has lasted this many seasons; before, they carry the previous one


def _adjusted(y: np.ndarray, season: Optional[int]) -> np.ndarray:
    """The series without its seasonal profile (when at least three seasons are available), so a season is not read as a regime."""
    y = np.asarray(y, dtype=float)
    if season and season >= 2 and len(y) >= 3 * season:
        positions = np.arange(len(y)) % season
        profile = np.array([np.nanmean(y[positions == k]) for k in range(season)])
        return y - profile[positions] + np.nanmean(y)
    return y


def _best_split(y: np.ndarray, sigma: float, minimum: int) -> Optional[tuple]:
    n = len(y)
    if n < 2 * minimum:
        return None
    csum = np.cumsum(y)
    k = np.arange(minimum, n - minimum + 1)
    left = csum[k - 1] / k
    right = (csum[-1] - csum[k - 1]) / (n - k)
    stat = np.abs(left - right) / (sigma * np.sqrt(1.0 / k + 1.0 / (n - k)))
    best = int(np.argmax(stat))
    return int(k[best]), float(stat[best])


def detect_breaks(y, season: Optional[int] = None, threshold: float = BREAK_THRESHOLD) -> List[int]:
    """Positions where a new regime starts, by binary segmentation of the level of ``y`` (missing values are skipped).

    Args:
        y: Training target in time order.
        season: Steps of the seasonal cycle to remove before looking at the level (``None`` for none).
        threshold: Standardised difference of means that makes a split.

    Returns:
        Sorted positions (indexes in ``y``) of the first row of each new regime, at most ``MAX_REGIMES - 1``.
    """
    values = _adjusted(np.asarray(y, dtype=float), season)
    ok = np.isfinite(values)
    index = np.flatnonzero(ok)
    series = values[ok]
    n = len(series)
    minimum = max(MIN_SEGMENT_ROWS, int(MIN_SEGMENT_SHARE * n), MIN_SEGMENT_SEASONS * (season or 0))
    if n < 4 * minimum:
        return []
    sigma = float(np.median(np.abs(np.diff(series) - np.median(np.diff(series)))) * 1.4826 / np.sqrt(2.0))
    if sigma <= 0:
        return []
    found: List[int] = []
    segments = [(0, n)]
    while segments and len(found) < MAX_REGIMES - 1:
        scored = []
        for start, stop in segments:
            split = _best_split(series[start:stop], sigma, minimum)
            if split is not None and split[1] > threshold:
                scored.append((split[1], start, stop, start + split[0]))
        if not scored:
            break
        _, start, stop, cut = max(scored)
        found.append(cut)
        segments = [s for s in segments if s != (start, stop)] + [(start, cut), (cut, stop)]
    return sorted(int(index[c]) for c in found)


def regime_ids(dates: pd.Series, break_dates: List[pd.Timestamp]) -> np.ndarray:
    """Regime number of each date: 0 before the first break, 1 from the first break on, and so on (the last one carries to later dates)."""
    d = pd.to_datetime(dates).to_numpy()
    if not break_dates:
        return np.zeros(len(d), dtype=float)
    return np.searchsorted(np.array(sorted(pd.to_datetime(break_dates).to_numpy())), d, side="right").astype(float)


def fit_regimes(df: pd.DataFrame, target: str, date_col: str, group_col: Optional[str], season: Optional[int]) -> Dict[str, Any]:
    """Break dates of the training target, one list for the series or one per group, and the regime the forecast rows carry.

    A regime that has lasted fewer than ``LAST_REGIME_SEASONS`` seasons at the end of the training rows is not trusted yet: the forecast rows
    carry the previous regime, so a level fitted on a short last segment is not extrapolated.

    Args:
        df: Training rows only.
        target: Target column.
        date_col: Date column.
        group_col: Group column, if any (each group has its own breaks).
        season: Steps of one season, to remove before the detection and to size the regimes.

    Returns:
        ``{'breaks': {group or None: [dates]}, 'carry': {group or None: regime number of the forecast rows}, 'end': last training date}``.
    """
    breaks: Dict[Any, List[pd.Timestamp]] = {}
    carry: Dict[Any, int] = {}
    parts = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]
    for key, part in parts:
        part = part.sort_values(date_col)
        found = [pd.Timestamp(part[date_col].iloc[p]) for p in detect_breaks(part[target].to_numpy(), season)]
        breaks[key] = found
        last_rows = int((pd.to_datetime(part[date_col]) >= found[-1]).sum()) if found else len(part)
        trusted = last_rows >= LAST_REGIME_SEASONS * (season or 5)
        carry[key] = len(found) if (trusted or not found) else len(found) - 1
    return {"breaks": breaks, "carry": carry, "end": pd.Timestamp(pd.to_datetime(df[date_col]).max())}


def add_regime_column(df: pd.DataFrame, date_col: str, group_col: Optional[str], fitted: Dict[str, Any], name: str = "regime") -> None:
    """Adds the regime number of each row (in place) from the break dates of the fit; rows after the training rows get the carried regime."""
    breaks, carry = fitted["breaks"], fitted.get("carry", {})
    dates = pd.to_datetime(df[date_col])
    ahead = (dates > fitted["end"]).to_numpy() if "end" in fitted else np.zeros(len(df), dtype=bool)
    if group_col and group_col in df.columns and None not in breaks:
        values = np.zeros(len(df), dtype=float)
        for key, found in breaks.items():
            rows = (df[group_col] == key).to_numpy()
            values[rows] = regime_ids(df.loc[rows, date_col], found)
            if key in carry:
                values[rows & ahead] = carry[key]
    else:
        found = next(iter(breaks.values())) if breaks else []
        values = regime_ids(df[date_col], found)
        if carry:
            values[ahead] = next(iter(carry.values()))
    df[name] = values
#: </regimes>
