# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Candidate group columns derived from the date column, for a series given without a group column.

A sub-daily series is often modelled as one sequence per period of the day (24, 48 or 96 groups, one step being one day): each group sees
one observation a day, so its lags, its adaptive windows and its Kalman state follow the daily rhythm. ``derived_group`` says which grouping the
dates allow and ``add_group`` writes the column. A daily series is split by day of week, one group a weekday, each group stepping by seven days
on its own phase (the data profiler keeps such groups as they are).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

#: <group_search_rules>
GROUP_COLUMN = "derived_group"
MAX_GROUPS = 96
MIN_CYCLES = 60


def derived_group(dates: pd.Series) -> Optional[Tuple[str, int]]:
    """The grouping the dates allow, or ``None``.

    Args:
        dates: The dates of the series (one row per date).

    Returns:
        ``("time_of_day", n)`` for a regular sub-daily step that divides the day into ``n`` periods (2 to ``MAX_GROUPS``), ``("day_of_week", 7)``
        for a daily step, ``None`` when the dates are not unique, the step is irregular or coarser than a day, the day would be cut in more
        than ``MAX_GROUPS`` periods, or there are too few cycles (``MIN_CYCLES`` days, four weeks for a daily step).
    """
    stamps = pd.to_datetime(pd.Series(dates)).dropna().drop_duplicates().sort_values()
    if len(stamps) != pd.Series(dates).notna().sum() or len(stamps) < 3:
        return None
    seconds = np.diff(stamps.to_numpy().astype("datetime64[s]").astype(np.int64))
    step = float(np.median(seconds))
    if step <= 0 or float(np.mean(seconds == step)) < 0.9:
        return None
    if 0.9 * 86400 <= step <= 1.1 * 86400:
        return ("day_of_week", 7) if len(stamps) >= 7 * 4 else None
    if step >= 86400:
        return None
    count = 86400.0 / step
    if abs(count - round(count)) > 1e-6 or not 2 <= round(count) <= MAX_GROUPS:
        return None
    return ("time_of_day", int(round(count))) if len(stamps) >= MIN_CYCLES * round(count) else None


def add_group(frame: pd.DataFrame, date_col: str, kind: str, count: int) -> pd.DataFrame:
    """A copy of ``frame`` with the group column ``GROUP_COLUMN`` (the period of the day, or the day of the week)."""
    out = frame.copy()
    stamps = pd.to_datetime(out[date_col])
    if kind == "day_of_week":
        out[GROUP_COLUMN] = stamps.dt.dayofweek.to_numpy()
    else:
        seconds = (stamps.dt.hour * 3600 + stamps.dt.minute * 60 + stamps.dt.second).to_numpy(dtype=float)
        out[GROUP_COLUMN] = np.minimum((seconds / (86400.0 / count)).astype(int), count - 1)
    return out


def sample_groups(frame: pd.DataFrame, group_col: str, target: str, size: int, by: str = "level") -> List:
    """The groups a search runs on: ``size`` of them, evenly spaced in the order of their level, and with ``by="level_volatility"`` half of
    them evenly spaced in the order of their volatility too (the coefficient of variation of the target), so that small, large, steady and
    erratic series are all in it. The result does not depend on the order of the rows (ties are broken by the group key).

    Args:
        frame: The training rows.
        group_col: Name of the group column.
        target: Name of the target column.
        size: Number of groups to take.
        by: ``"level"`` or ``"level_volatility"``.

    Returns:
        The groups taken, in the order of their level. All the groups when there are not more than ``size``.
    """
    if by not in ("level", "level_volatility"):
        raise ValueError(f"search_sample must be 'level' or 'level_volatility'; got {by!r}")
    stats = frame.groupby(group_col)[target].agg(["mean", "std"])
    keys = sorted(stats.index, key=str)
    stats = stats.loc[keys]
    if len(stats) <= size:
        return list(stats.sort_values("mean", kind="mergesort").index)
    by_level = list(stats.sort_values("mean", kind="mergesort").index)

    def spaced(order, count):
        return [order[i] for i in np.linspace(0, len(order) - 1, count).round().astype(int)]

    if by == "level":
        chosen = spaced(by_level, size)
    else:
        spread = (stats["std"].fillna(0.0) / (stats["mean"].abs() + 1e-9)).sort_values(kind="mergesort")
        by_volatility = list(spread.index)
        chosen = list(dict.fromkeys(spaced(by_level, (size + 1) // 2) + spaced(by_volatility, size // 2)))
        count = size
        while len(chosen) < size and count < len(stats):   # a group in both orders leaves room: fill with the next positions of the level order
            count += 1
            chosen = list(dict.fromkeys(chosen + spaced(by_level, count)))[:size]
    chosen = set(chosen)
    return [g for g in by_level if g in chosen]
#: </group_search_rules>
