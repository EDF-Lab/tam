# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Seasonal anchor of the AutoTAM search (opt-in: ``AutoTAM(..., seasonal_anchor=True)``).

The anchor is an expert that regresses the target on its value one season earlier, so that the pool always holds the "last season repeated" reference
(calibrated by an intercept and a slope) and the ensembles can fall back on it.
"""

import re
from typing import List, Optional

import pandas as pd

#: <anchor_rules>
ANCHOR_ISLAND = "Anchor"


def natural_season(freq: str) -> Optional[int]:
    """Number of steps of the natural season of a sampling step, or ``None`` when there is none.

    A day for a step under a day (24 for hourly data), a week for daily data (7), a year for weekly (52) and monthly (12) data, four quarters (4).

    Args:
        freq: Pandas frequency alias of the data, as recorded by the data profiler.

    Returns:
        The season in steps, or ``None`` (yearly data, a step of several days, an unknown alias).
    """
    alias = re.sub(r"(\d*)T$", lambda m: f"{m.group(1)}min", str(freq))
    alias = re.sub(r"(\d*)H$", lambda m: f"{m.group(1)}h", alias)
    try:
        offset = pd.tseries.frequencies.to_offset(alias)
    except ValueError:
        return None
    if isinstance(offset, pd.offsets.Day):
        return 7 if offset.n == 1 else None
    if isinstance(offset, pd.offsets.Week):
        return 52 if offset.n == 1 else None
    if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return 12 if offset.n == 1 else None
    if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return 4 if offset.n == 1 else None
    if isinstance(offset, pd.offsets.Tick):
        seconds = offset.nanos / 1e9
        if 0 < seconds < 86400 and 86400 % seconds == 0:
            return int(86400 // seconds)
    return None


def resolve_season(season: Optional[int], freq: str, target_lags: List[int]) -> int:
    """The season of the anchor: the explicit one, else the natural season of the sampling step; validated against the horizon.

    Args:
        season: The ``season=`` argument, or ``None``.
        freq: Pandas frequency alias of the data.
        target_lags: Lag orders of the target; the smallest is the horizon.

    Returns:
        The season in steps.

    Raises:
        ValueError: ``season`` is not a positive integer, the step has no natural season, or the season is shorter than the horizon (the value
            one season earlier would not be known at the forecast origin).
    """
    if season is None:
        season = natural_season(freq)
        if season is None:
            raise ValueError(f"The sampling step '{freq}' has no natural season: give season= (the number of steps of one season).")
    elif isinstance(season, bool) or not isinstance(season, int) or season < 1:
        raise ValueError(f"season must be a positive integer number of steps, got {season!r}.")
    horizon = min(target_lags) if target_lags else 1
    if season < horizon:
        raise ValueError(
            f"The season of the anchor ({season} steps) is shorter than the horizon ({horizon}, the smallest lag of the target): "
            "the value one season earlier would not be known at the forecast origin."
        )
    return int(season)


def anchor_column(season: int) -> str:
    """Name of the column holding the target one season earlier."""
    return f"anchor_lag_{season}"


def anchor_formula(target: str, season: int) -> str:
    """Formula of the anchor expert: the target regressed on its value one season earlier."""
    return f"{target} ~ l({anchor_column(season)})"


def add_anchor_lag(df: pd.DataFrame, target: str, season: int, group_col: Optional[str]) -> None:
    """Adds the target one season earlier, inside each group, the way the lags of the target are built (in place)."""
    if group_col and group_col in df.columns:
        df[anchor_column(season)] = df.groupby(group_col)[target].shift(season)
    else:
        df[anchor_column(season)] = df[target].shift(season)
#: </anchor_rules>
