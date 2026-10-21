# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Columns that are not ordinary covariates: an exposure offset, a pair of spatial coordinates, an identifier with many levels.

The feature engineer divides the first two numeric columns by each other and smooths every numeric column over time. For an exposure it makes the
offset a feature of a tensor product; for ``lon`` and ``lat`` it makes ``lon_div_lat``, which is neither translation nor rotation invariant; for an
identifier with fifty levels it makes a Fourier series over the identifier numbers. This module names these columns so that the engineer leaves them
alone and the search offers what they are for: ``offset(col)`` for an exposure, a thin-plate or tensor surface for a coordinate pair, a ridge
``c(id)`` and a factor smooth ``te(s(x), c(id))`` for an identifier.
"""

import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

#: <structure_columns>
SPATIAL_NAMES: Tuple[Tuple[str, str], ...] = (
    ("lon", "lat"), ("longitude", "latitude"), ("lo", "la"), ("lng", "lat"), ("x", "y"), ("e", "n"), ("easting", "northing"))
ID_NAME = re.compile(r"(^|_|\.)(id|ids|subj|subject|plant|tree|traj|plot|station|speaker|word|region|district|site|ship|patient|animal|pig)(_|\.|\d|$)",
                     re.IGNORECASE)
MIN_ID_LEVELS = 16
MAX_ID_LEVELS = 250
MAX_STRING_ID_LEVELS = 500


def spatial_pairs(columns: Iterable[str]) -> List[Tuple[str, str]]:
    """The pairs of coordinate columns among ``columns`` (case-insensitive: ``Lon``/``Lat``, ``X``/``Y``, ``E``/``N``), first coordinate first."""
    by_lower = {str(c).lower(): c for c in columns}
    return [(by_lower[a], by_lower[b]) for a, b in SPATIAL_NAMES if a in by_lower and b in by_lower]


def with_offset(formula: str, column: Optional[str]) -> str:
    """``formula`` with ``+ offset(column)`` on its right-hand side (unchanged when it has one already or ``column`` is ``None``)."""
    if not column or re.search(rf"offset\s*\(\s*{re.escape(column)}\s*\)", formula):
        return formula
    if "~" not in formula:
        return formula
    left, right = formula.split("~", 1)
    return f"{left.rstrip()} ~ offset({column})" if right.strip() in ("", "1") else f"{formula.rstrip()} + offset({column})"


def random_effect_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    """The identifier columns of ``frame``: many levels that repeat.

    A column is an identifier when it is text or categorical with 16 to 500 levels (at most half the rows), or whole numbers with 16 to 250
    levels that each repeat at least twice on average and either carry an identifier word in their name (``id``, ``subj``, ``tree``, ``site``...)
    or have exactly the same count in every level (a balanced panel).
    """
    skip = set(exclude)
    rows = len(frame)
    found: List[str] = []
    for column in frame.columns:
        if column in skip:
            continue
        values = frame[column].dropna()
        levels = int(values.nunique())
        if values.empty or levels < MIN_ID_LEVELS or levels > rows // 2:
            continue
        if not pd.api.types.is_numeric_dtype(values):
            if levels <= MAX_STRING_ID_LEVELS:
                found.append(column)
            continue
        if levels > MAX_ID_LEVELS or not bool((values % 1 == 0).all()) or rows / levels < 2.0:
            continue
        counts = values.value_counts()
        if ID_NAME.search(str(column)) or counts.max() == counts.min():   # an id word in the name, or a perfectly balanced panel
            found.append(column)
    return found
#: </structure_columns>
