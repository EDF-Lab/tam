# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Missing-data codes and impossible readings, masked and imputed before the lags are built.

A station that writes -9999 where it has no reading, a sensor that reports -273 degrees: fitted as values they dominate a squared loss, and a lag
carries them forward one step, one day, one week later. ``SentinelCleaner`` learns on the training rows which values of which columns are not
readings (repeated round negative codes, the rare negatives of a quantity that is never negative, readings outside the physical range of a
temperature or of a wind direction), replaces them by missing values and fills those with the median of the same position of the season (hour of
the day, day of the week, month) of the training rows, in the same group. Only the cells it masked are filled: a missing target to forecast stays
missing. What it learned is applied unchanged to the rows to forecast.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from tam.model.autotam.floor import season_position

#: <sentinel_cleaning_rules>
MIN_COUNT = 2
MIN_SHARE = 0.001
ROUND_CODES = (-99.0, -999.0, -9999.0, -99999.0)
NEGATIVE_SHARE_LIMIT = 0.005      # a quantity that is never negative: at most this share of negatives is read as codes
NEGATIVE_QUANTITY_SHARE = 0.99    # ... when at least this share of the readings is not negative
TEMPERATURE_NAME = re.compile(r"(temp|tmp)", re.IGNORECASE)
DIRECTION_NAME = re.compile(r"(wdir|wind_?dir|direction|azimuth)", re.IGNORECASE)
CELSIUS_RANGE = (-60.0, 60.0)
DIRECTION_RANGE = (0.0, 360.0)


def _is_round_code(value: float) -> bool:
    return value in ROUND_CODES or (value <= -200.0 and value % 100.0 == 0.0)


def find_codes(values: np.ndarray) -> List[float]:
    """The round negative constants that repeat in ``values`` (-9999, -999, -200 ...), in increasing order."""
    finite = values[np.isfinite(values)]
    if finite.size < 5:
        return []
    needed = max(MIN_COUNT, int(np.ceil(MIN_SHARE * finite.size)))
    counts = pd.Series(finite[finite <= -99.0]).value_counts()
    return sorted(float(v) for v, c in counts.items() if c >= needed and _is_round_code(float(v)))


def physical_range(name: str, values: np.ndarray, given: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """The range a reading of this column can take, or ``None``: the one given, a temperature in degrees Celsius (a column whose name says
    temperature and whose median lies inside the range, so a Kelvin series is left alone) or a wind direction in degrees."""
    if given is not None:
        return float(given[0]), float(given[1])
    finite = values[np.isfinite(values)]
    if finite.size < 5:
        return None
    median = float(np.median(finite))
    if TEMPERATURE_NAME.search(name) and CELSIUS_RANGE[0] <= median <= CELSIUS_RANGE[1]:
        return CELSIUS_RANGE
    if DIRECTION_NAME.search(name) and DIRECTION_RANGE[0] <= median <= DIRECTION_RANGE[1]:
        return DIRECTION_RANGE
    return None


class SentinelCleaner:
    """Learns the codes and ranges of the columns on the training rows, then masks and imputes them in any frame."""

    def __init__(self, columns: Iterable[str], date_col: Optional[str] = None, group_col: Optional[str] = None,
                 physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None, negative_columns: Iterable[str] = ()):
        self.columns = list(columns)
        self.negative_columns = set(negative_columns)   # the columns that are a quantity never negative: the target
        self.date_col = date_col
        self.group_col = group_col
        self.physical_bounds = dict(physical_bounds or {})
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.medians: Dict[str, Any] = {}
        self.steps_per_day: Optional[float] = None

    def _positions(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.date_col or self.date_col not in frame.columns or self.steps_per_day is None:
            return np.zeros(len(frame), dtype=int)
        try:
            return season_position(pd.to_datetime(frame[self.date_col]), self.steps_per_day)[0]
        except ValueError:
            return np.zeros(len(frame), dtype=int)

    def _flags(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        rule = self.rules.get(column)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not rule:
            return np.zeros(len(frame), dtype=bool)
        flagged = np.isin(values, rule["codes"])
        if rule["bounds"] is not None:
            flagged |= np.isfinite(values) & ((values < rule["bounds"][0]) | (values > rule["bounds"][1]))
        if rule["negatives"]:
            flagged |= np.isfinite(values) & (values < 0.0)
        return flagged

    def fit(self, frame: pd.DataFrame) -> "SentinelCleaner":
        """Learns the rules of every numeric column of ``frame`` (the training rows, all of them) and the seasonal medians of what is left."""
        if self.date_col and self.date_col in frame.columns:
            stamps = pd.to_datetime(frame[self.date_col]).drop_duplicates().sort_values()
            seconds = stamps.diff().dt.total_seconds().dropna()
            seconds = seconds[seconds > 0]
            self.steps_per_day = 86400.0 / float(seconds.median()) if len(seconds) else None
        for column in self.columns:
            if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size < 5:
                continue
            codes = find_codes(values)
            bounds = physical_range(column, values, self.physical_bounds.get(column))
            kept = finite[~np.isin(finite, codes)]
            negatives = False
            if column in self.negative_columns and kept.size and not codes and bounds is None:
                share = float(np.mean(kept < 0.0))
                negatives = 0.0 < share <= NEGATIVE_SHARE_LIMIT and (1.0 - share) >= NEGATIVE_QUANTITY_SHARE and float(kept[kept >= 0].min()) >= 0.0
            if codes or bounds is not None or negatives:
                self.rules[column] = {"codes": codes, "bounds": bounds, "negatives": negatives}
        positions = self._positions(frame)
        groups = frame[self.group_col].to_numpy() if self.group_col and self.group_col in frame.columns else np.zeros(len(frame), dtype=int)
        for column in self.rules:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(values) & ~self._flags(frame, column)
            table = pd.DataFrame({"g": groups[ok], "p": positions[ok], "v": values[ok]})
            self.medians[column] = {"by": table.groupby(["g", "p"])["v"].median().to_dict(), "all": float(np.median(values[ok])) if ok.any() else 0.0}
        return self

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """A copy of ``frame`` where the flagged cells are replaced by the seasonal median of the training rows of the same group."""
        if not self.rules:
            return frame
        out = frame.copy()
        positions = self._positions(out)
        groups = out[self.group_col].to_numpy() if self.group_col and self.group_col in out.columns else np.zeros(len(out), dtype=int)
        for column in self.rules:
            if column not in out.columns:
                continue
            flagged = self._flags(out, column)
            if not flagged.any():
                continue
            table = self.medians[column]
            fill = np.array([table["by"].get((g, p), table["all"]) for g, p in zip(groups[flagged], positions[flagged])], dtype=float)
            values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float).copy()
            values[flagged] = fill
            out[column] = values
        return out

    def report(self, frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Per column: the codes, the range, whether negatives were read as codes, and how many cells of ``frame`` are flagged."""
        return {c: {"codes": r["codes"], "bounds": r["bounds"], "negatives": r["negatives"],
                    "masked": int(self._flags(frame, c).sum()) if c in frame.columns else 0} for c, r in self.rules.items()}
#: </sentinel_cleaning_rules>
