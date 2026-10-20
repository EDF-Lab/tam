# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Extrapolation guards of the AutoTAM search (opt-in: ``AutoTAM(..., extrapolation_guards=True)``).

The worst forecasts of a search are extrapolations: a linear term on a counter beyond the history, a neural or PID term fitted on a few
hundred rows, a formula with as many coefficients as rows. These rules keep such candidates out of the champions and bound the forecast.
"""

import re
from typing import Callable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from tam.common.utils import parse_formula_to_terms

#: <guards_constants>
GUARD_MARGIN_FRACTION = 0.15
GUARD_MAX_COEFFICIENT_FRACTION = 0.5
GUARD_MIN_ROWS_FOR_NEURAL_AND_PID = 1000
GUARD_MONOTONE_SHARE = 0.99
#: </guards_constants>


#: <guards_rules>
def guard_bounds(target_min: float, target_max: float, margin_fraction: float = GUARD_MARGIN_FRACTION) -> Tuple[float, float]:
    """Bounds a forecast must stay in: the training range plus a margin, floored at zero for a target that was never negative.

    Args:
        target_min: Smallest value of the training target.
        target_max: Largest value of the training target.
        margin_fraction: Margin on each side, as a fraction of the range.

    Returns:
        The ``(low, high)`` bounds.
    """
    margin = margin_fraction * (target_max - target_min)
    low, high = target_min - margin, target_max + margin
    if target_min >= 0.0:
        low = max(low, 0.0)
    return float(low), float(high)


def monotone_features(df: pd.DataFrame, columns, share: float = GUARD_MONOTONE_SHARE) -> Set[str]:
    """Numeric columns that are counters: non-decreasing (or non-increasing) on at least ``share`` of the steps and not constant.

    Args:
        df: Training frame, in time order.
        columns: Columns to examine.
        share: Share of the steps that must go in the same direction.

    Returns:
        The names of the monotone columns.
    """
    found = set()
    for column in columns:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        steps = np.diff(df[column].dropna().to_numpy(dtype=float))
        if steps.size < 3 or not np.any(steps != 0):
            continue
        if (steps >= 0).mean() >= share or (steps <= 0).mean() >= share:
            found.add(column)
    return found


def guard_violation(formula: str, n_rows: int, monotone: Set[str], complexity: Callable[[str], int]) -> Optional[str]:
    """Why a candidate formula must not be a champion, or ``None`` when it may.

    Args:
        formula: Candidate formula.
        n_rows: Rows of the fit set.
        monotone: Counters of the data (see ``monotone_features``).
        complexity: Function giving the number of coefficients of a formula.

    Returns:
        The reason, written ``<code>: <explanation>`` with the code ``coefficients``, ``rows`` or ``counter``, or ``None``.
    """
    bound = GUARD_MAX_COEFFICIENT_FRACTION * n_rows
    k = complexity(formula)
    if k > bound:
        return f"coefficients: {k} coefficients for {n_rows} rows (bound {bound:.0f})"
    try:
        _, terms = parse_formula_to_terms(formula)
    except ValueError:
        return None
    for term in terms:
        kind, feature = term.get("type"), term.get("feature")
        if kind in ("n", "pid") and n_rows < GUARD_MIN_ROWS_FOR_NEURAL_AND_PID:
            return f"rows: a {kind}() term needs at least {GUARD_MIN_ROWS_FOR_NEURAL_AND_PID} rows, there are {n_rows}"
        if kind == "l" and feature in monotone:
            return f"counter: a linear term on the counter '{feature}' extrapolates without bound"
    return None
#: </guards_rules>
