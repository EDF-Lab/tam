# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Series that are not running: forecast zero instead of a level fitted on other series.

A shop that has not opened yet, a cluster that is switched off, a store closed for works, the night of a solar plant: a model fits a level for
them from the series around, and a seasonal naive benchmark of an all-zero series has a scale of about one, so a forecast of forty is a score
of forty. ``InactivityGate`` reads from the training rows which groups never had a non-zero target (unopened), which positions of the season
are always zero (the night), and from the rows to forecast which are declared closed by a known-future column (``Open == 0``), and sets the
forecast of those rows to zero, point and quantiles. ``drop_leading_zeros`` takes the rows before the first non-zero value of each group out of
the fit, so that a shop opened last month is not estimated from years of inactivity.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from tam.model.autotam.floor import season_position

#: <inactivity_gate>
ZERO_POSITION_SHARE = 0.95   # a position of the season is "always zero" when this share of its training values is zero
MIN_POSITION_ROWS = 20


def drop_leading_zeros(frame: pd.DataFrame, target: str, group_col: Optional[str], date_col: Optional[str]) -> pd.DataFrame:
    """``frame`` without the rows of each group dated before its first non-zero target (a group that is all zero is kept as it is)."""
    if frame is None or target not in frame.columns:
        return frame
    values = pd.to_numeric(frame[target], errors="coerce")
    nonzero = (values.notna() & (values != 0)).to_numpy()
    if date_col and date_col in frame.columns:
        order = pd.to_datetime(frame[date_col]).reset_index(drop=True)
    else:
        order = pd.Series(np.arange(len(frame)))
    keys = (frame[group_col] if group_col and group_col in frame.columns else pd.Series(0, index=frame.index)).reset_index(drop=True)
    first = order[nonzero].groupby(keys[nonzero]).min()
    start = keys.map(first)
    keep = (start.isna() | (order >= start)).to_numpy()
    return frame if keep.all() else frame[keep]


class InactivityGate:
    """Learns the unopened groups and the zero positions of the training rows, then tells which rows of a frame are forecast zero."""

    def __init__(self, target: str, date_col: Optional[str], group_col: Optional[str], closed_when: Optional[Dict[str, float]] = None):
        self.target, self.date_col, self.group_col = target, date_col, group_col
        self.closed_when = dict(closed_when or {})
        self.unopened: Set[Any] = set()
        self.zero_positions: Set[Tuple[Any, int]] = set()
        self.inactive_at_cutoff: List[Any] = []
        self.steps_per_day: Optional[float] = None

    def _keys(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.group_col].to_numpy() if self.group_col and self.group_col in frame.columns else np.zeros(len(frame), dtype=int)

    def _positions(self, frame: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.date_col or self.date_col not in frame.columns or not self.steps_per_day:
            return None
        try:
            return season_position(pd.to_datetime(frame[self.date_col]), self.steps_per_day)[0]
        except ValueError:
            return None

    def fit(self, rows: pd.DataFrame, cutoff_rows: int = 12) -> "InactivityGate":
        """Learns from the training rows (in time order inside each group)."""
        if self.date_col and self.date_col in rows.columns:
            stamps = pd.to_datetime(rows[self.date_col]).drop_duplicates().sort_values()
            seconds = stamps.diff().dt.total_seconds().dropna()
            seconds = seconds[seconds > 0]
            self.steps_per_day = 86400.0 / float(seconds.median()) if len(seconds) else None
        y = pd.to_numeric(rows[self.target], errors="coerce").to_numpy(dtype=float)
        keys = self._keys(rows)
        positions = self._positions(rows)
        for key in pd.unique(keys):
            mine = keys == key
            seen = y[mine][np.isfinite(y[mine])]
            if seen.size and not np.any(seen != 0):
                self.unopened.add(key)
                continue
            if seen.size and not np.any(seen[-cutoff_rows:] != 0):
                self.inactive_at_cutoff.append(key)
            if positions is not None and self.steps_per_day and self.steps_per_day > 1.5:     # a profile within the day: the night of a solar plant
                table = pd.DataFrame({"p": positions[mine], "zero": (y[mine] == 0), "ok": np.isfinite(y[mine])})
                share = table[table["ok"]].groupby("p")["zero"].agg(["mean", "size"])
                for position, row in share.iterrows():
                    if row["size"] >= MIN_POSITION_ROWS and row["mean"] >= ZERO_POSITION_SHARE:
                        self.zero_positions.add((key, int(position)))
        return self

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        """True on the rows to forecast as zero: a group that never opened, a position that is always zero, a row declared closed."""
        keys = self._keys(frame)
        out = np.isin(keys, list(self.unopened)) if self.unopened else np.zeros(len(frame), dtype=bool)
        if self.zero_positions:
            positions = self._positions(frame)
            if positions is not None:
                out = out | np.array([(k, int(p)) in self.zero_positions for k, p in zip(keys, positions)], dtype=bool)
        for column, closed_value in self.closed_when.items():
            if column in frame.columns:
                out = out | (pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) == float(closed_value))
        return out

    def apply(self, values: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        """``values`` (one row per row of ``frame``, any number of forecast columns) with the gated rows set to zero."""
        rows = self.mask(frame)
        if not rows.any() or len(rows) != len(values):
            return values
        out = values.copy()
        numeric = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        out.loc[rows, numeric] = 0.0
        return out
#: </inactivity_gate>
