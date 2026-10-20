# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Forecasts made outside ``AutoTAM`` (the caller's rules model, a baseline) as members of its Apex aggregation.

The caller does not know which rows ``fit()`` keeps for validation, so an external forecast is given by key: a ``pd.Series`` indexed by the date
(or by ``(date, group)``), or a ``pd.DataFrame`` with the date column, the group column when there is one, and a ``forecast`` column. The
forecast covers the validation rows at ``fit()`` and the rows to forecast at ``predict()``; ``AutoTAM`` looks the rows up by key and never
refits it.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

#: <external_experts>
VALUE_COLUMN = "forecast"
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def normalise_external(spec: Optional[Dict[str, Any]], date_col: Optional[str], group_col: Optional[str], taken: List[str] = ()) -> Dict[str, pd.DataFrame]:
    """``{name: frame of the key columns and VALUE_COLUMN}`` for the ``external_experts`` argument.

    Raises:
        ValueError: No date column, a name that is not safe inside a formula (letters, digits and underscores, starting with a letter) or that is
            already an expert, or a forecast that is not a series or a frame in the forms above.
    """
    if not spec:
        return {}
    if not date_col:
        raise ValueError("external_experts needs the date column: the forecasts are matched to the rows by date (and group).")
    out: Dict[str, pd.DataFrame] = {}
    for name, forecast in spec.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"external expert name {name!r}: use letters, digits and underscores, starting with a letter (it is written into a formula).")
        if name in taken:
            raise ValueError(f"external expert name {name!r} is already the name of an expert.")
        if isinstance(forecast, pd.DataFrame):
            if date_col not in forecast.columns or VALUE_COLUMN not in forecast.columns:
                raise ValueError(f"external expert {name!r}: a frame needs the columns {date_col!r} and {VALUE_COLUMN!r}.")
            keys = [date_col] + ([group_col] if group_col and group_col in forecast.columns else [])
            frame = forecast[keys + [VALUE_COLUMN]].copy()
        elif isinstance(forecast, pd.Series):
            levels = forecast.index.nlevels
            if levels == 1:
                frame = pd.DataFrame({date_col: forecast.index, VALUE_COLUMN: forecast.to_numpy()})
            elif levels == 2 and group_col:
                frame = pd.DataFrame({date_col: forecast.index.get_level_values(0), group_col: forecast.index.get_level_values(1),
                                      VALUE_COLUMN: forecast.to_numpy()})
            else:
                raise ValueError(f"external expert {name!r}: a series is indexed by the date, or by (date, group) when there is a group column.")
        else:
            raise ValueError(f"external expert {name!r}: give a pd.Series indexed by date (or (date, group)) or a pd.DataFrame.")
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame[VALUE_COLUMN] = pd.to_numeric(frame[VALUE_COLUMN], errors="coerce")
        out[name] = frame.drop_duplicates(subset=[c for c in frame.columns if c != VALUE_COLUMN], keep="last")
    return out


def look_up(rows: pd.DataFrame, forecast: pd.DataFrame, date_col: str) -> np.ndarray:
    """The values of ``forecast`` on the rows of ``rows``, in their order (NaN where the key is missing). Keys: the date and the group
    column when the forecast has one."""
    keys = [c for c in forecast.columns if c != VALUE_COLUMN]
    left = rows[[c for c in keys if c in rows.columns]].copy()
    left[date_col] = pd.to_datetime(left[date_col])
    on = list(left.columns)
    left["__row__"] = np.arange(len(left))
    merged = left.merge(forecast[on + [VALUE_COLUMN]], on=on, how="left").sort_values("__row__")
    return merged[VALUE_COLUMN].to_numpy(dtype=float)
#: </external_experts>
