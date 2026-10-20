# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
A guard per series: a series for which a seasonal baseline was better than the final model on the validation rows is forecast by that baseline.

The pooled floor and the seed floor bound the final model from below over all the rows together. A panel of hundreds of series hides the series
for which the model loses to the seasonal profile or to the last season repeated, so the guard works series by series: for each group, the
final model is replaced by the best guard expert (the experts of the ``Floor`` and ``Anchor`` islands) when that expert has the smaller RMSE
on the validation rows of the group.
"""

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from tam.model.autotam.anchor import ANCHOR_ISLAND
from tam.model.autotam.floor import FLOOR_ISLAND

#: <series_guard>
GUARD_ISLANDS = (FLOOR_ISLAND, ANCHOR_ISLAND)
SINGLE_SERIES = "__series__"


def guarded_series(truth: np.ndarray, final: np.ndarray, guards: Dict[str, np.ndarray], groups: Optional[Iterable[Any]]) -> Dict[Any, str]:
    """``{group: guard expert}`` for the groups on which a guard expert has the smaller RMSE than the final model.

    Args:
        truth: The target on the validation rows.
        final: The forecast of the final model on the same rows.
        guards: ``{expert name: forecast}`` of the guard experts on the same rows.
        groups: The group of each row; ``None`` for a single series (the key is ``SINGLE_SERIES``).

    Returns:
        Only the groups to guard. Rows where the truth or a forecast is missing do not count; a group where the final model has no finite error is
        not guarded.
    """
    keys = np.asarray([SINGLE_SERIES] * len(truth) if groups is None else list(groups), dtype=object)
    out: Dict[Any, str] = {}
    for key in pd.unique(keys):
        rows = keys == key
        errors = {}
        for name, forecast in [("__final__", final), *guards.items()]:
            gap = np.asarray(truth, dtype=float)[rows] - np.asarray(forecast, dtype=float)[rows]
            gap = gap[np.isfinite(gap)]
            if gap.size:
                errors[name] = float(np.sqrt(np.mean(gap ** 2)))
        if "__final__" not in errors:
            continue
        others = {name: error for name, error in errors.items() if name != "__final__"}
        if others:
            best = min(others, key=lambda name: (others[name], name))
            if others[best] < errors["__final__"]:
                out[key] = best
    return out


def apply_guard(frame: pd.DataFrame, column: str, guarded: Dict[Any, str], groups: Optional[Iterable[Any]]) -> pd.DataFrame:
    """``frame`` with ``column`` replaced, on the rows of each guarded group, by the column of its guard expert (where that one is finite)."""
    if not guarded or column not in frame.columns:
        return frame
    keys = np.asarray([SINGLE_SERIES] * len(frame) if groups is None else list(groups), dtype=object)
    values = frame[column].to_numpy(dtype=float).copy()
    for key, expert in guarded.items():
        if expert not in frame.columns:
            continue
        rows = (keys == key) & np.isfinite(frame[expert].to_numpy(dtype=float))
        values[rows] = frame[expert].to_numpy(dtype=float)[rows]
    out = frame.copy()
    out[column] = values
    return out
#: </series_guard>
