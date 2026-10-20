# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Differenced targets: the model learns ``d_t = y_t - y_{t-h}`` and the forecast adds the level back, step by step.

For a cumulative or monotone series, or a unit root, the level is the wrong thing to model: it has no stable mean and a model of the level
extrapolates a trend. The increment ``d`` over the forecast horizon ``h`` is stable. The training frames get ``d`` as their target (the
level is kept apart, never as a column the model could read); a forecast row takes the level ``h`` steps before it, observed or already
forecast, so the levels are rebuilt recursively inside the horizon and the errors of the increments add up only once per ``h`` steps.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: <differencing>
LEVEL = "__level__"
KEY = "__key__"


def _keys(df: pd.DataFrame, group_col: Optional[str]) -> pd.Series:
    if group_col and group_col in df.columns:
        return df[group_col].map(str)       # a string key, so a missing group is a group of its own and pandas never drops its rows
    return pd.Series(["__all__"] * len(df), index=df.index, dtype=object)


def difference_training(frames: Dict[str, Optional[pd.DataFrame]], target: str, date_col: str, group_col: Optional[str],
                        h: int) -> Tuple[Dict[str, Optional[pd.DataFrame]], pd.DataFrame]:
    """The training frames with ``y_t - y_{t-h}`` as their target, and the levels kept apart.

    Args:
        frames: Named training frames (``df_train`` or ``df_fit``, ``df_dev``, ``df_val``), in time order; ``None`` frames are left out.
        target: Name of the target.
        date_col: Name of the date column (the order inside each group).
        group_col: Name of the group column, if any (the lag is taken inside each group).
        h: The lag of the difference, in rows.

    Returns:
        The frames, without the ``h`` first rows of each group (they have no difference), and a frame of the levels (``__key__``, date, level)
        of all the training rows, for the add-back.

    Raises:
        ValueError: ``h`` is not a positive integer, or there is no date column.
    """
    if int(h) != h or h < 1:
        raise ValueError(f"difference must be a positive whole number of steps, got {h!r}.")
    if not date_col:
        raise ValueError("difference needs a date column: pass date_col=.")
    parts = []
    for name, frame in frames.items():
        if frame is None:
            continue
        part = frame[[c for c in (date_col, group_col, target) if c and c in frame.columns]].copy()
        part["__frame__"], part["__row__"] = name, np.arange(len(frame))
        parts.append(part)
    allf = pd.concat(parts, ignore_index=True)
    allf[KEY] = _keys(allf, group_col)
    allf["__label__"] = allf[KEY]
    allf[date_col] = pd.to_datetime(allf[date_col])
    allf = allf.sort_values(["__label__", date_col], kind="mergesort").reset_index(drop=True)
    previous = allf.groupby("__label__", sort=False)[target].shift(h)
    allf["__d__"] = allf[target] - previous
    out: Dict[str, Optional[pd.DataFrame]] = {}
    for name, frame in frames.items():
        if frame is None:
            out[name] = None
            continue
        mine = allf[allf["__frame__"] == name].sort_values("__row__")
        new = frame.copy()
        new[target] = mine["__d__"].to_numpy()
        out[name] = new[new[target].notna()]
    levels = allf[[KEY, date_col, target]].rename(columns={target: LEVEL})
    return out, levels


def _combined(df_test: pd.DataFrame, levels: pd.DataFrame, target: str, date_col: str, group_col: Optional[str]) -> pd.DataFrame:
    """Training levels and the rows of the test frame, in time order inside each group.

    A test row on a date the training rows already have takes the training level (the model was fitted on it) and replaces that training row.
    """
    test = pd.DataFrame({KEY: _keys(df_test, group_col).to_numpy(), date_col: pd.to_datetime(df_test[date_col]).to_numpy(),
                         LEVEL: df_test[target].to_numpy(dtype=float) if target in df_test.columns else np.nan})
    test["__test__"] = np.arange(len(test))
    train = levels.copy()
    train[date_col] = pd.to_datetime(train[date_col])
    train["__test__"] = -1
    both = pd.concat([train, test], ignore_index=True)
    both["__order__"] = (both["__test__"] >= 0).astype(int)
    both["__label__"] = both[KEY]
    both = both.sort_values(["__label__", date_col, "__order__"], kind="mergesort")
    twin = both.duplicated(subset=["__label__", date_col], keep=False)
    both[LEVEL] = both.groupby(["__label__", date_col], sort=False)[LEVEL].transform("first").where(twin, both[LEVEL])
    both = both[~(twin & (both["__test__"] < 0))]
    return both.reset_index(drop=True)


def difference_test(df_test: pd.DataFrame, levels: pd.DataFrame, target: str, date_col: str, group_col: Optional[str], h: int) -> pd.DataFrame:
    """The test frame with the increment as its target (missing where the level or the level ``h`` steps before is not observed)."""
    both = _combined(df_test, levels, target, date_col, group_col)
    both["__d__"] = both[LEVEL] - both.groupby("__label__", sort=False)[LEVEL].shift(h)
    mine = both[both["__test__"] >= 0].sort_values("__test__")
    out = df_test.copy()
    out[target] = mine["__d__"].to_numpy()
    return out


def add_back(predicted: pd.DataFrame, df_test: pd.DataFrame, levels: pd.DataFrame, target: str, date_col: str, group_col: Optional[str],
             h: int) -> pd.DataFrame:
    """The forecasts of the increments turned into forecasts of the level, column by column.

    A row takes the level ``h`` steps before it: observed (in the training rows or in the test frame) or, inside the forecast, the level the
    column forecast for that row. The columns are processed together, row after row, so the recursion is vectorised over the models.

    Args:
        predicted: One row per row of ``df_test`` (the same order) and one column per model, holding forecast increments.
        df_test: The frame that was forecast (its target may be missing on the forecast rows).
        levels: The training levels returned by ``difference_training``.
        target: Name of the target.
        date_col: Name of the date column.
        group_col: Name of the group column, if any.
        h: The lag of the difference.

    Returns:
        A frame like ``predicted`` with the levels.
    """
    both = _combined(df_test, levels, target, date_col, group_col)
    matrix = predicted.to_numpy(dtype=float)
    width = matrix.shape[1]
    out = np.full_like(matrix, np.nan)
    for _, group in both.groupby("__label__", sort=False):
        known = group[LEVEL].to_numpy(dtype=float)
        test_row = group["__test__"].to_numpy()
        current = np.full((len(group), width), np.nan)       # the level of each step, observed or forecast, that later steps build on
        for i in range(len(group)):
            if test_row[i] < 0:
                current[i] = known[i]
                continue
            base = current[i - h] if i - h >= 0 else np.full(width, np.nan)
            forecast = base + matrix[test_row[i]]
            out[test_row[i]] = forecast
            current[i] = known[i] if np.isfinite(known[i]) else forecast
    return pd.DataFrame(out, index=predicted.index, columns=predicted.columns)
#: </differencing>
