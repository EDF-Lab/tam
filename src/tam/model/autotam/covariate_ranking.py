# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Covariates ranked by how much they say about the target, before the search draws among them.

A search that picks its covariates by position or by chance wastes its budget on the ones that say nothing: the temperature of a heating load
ranked after a radiation column, the wind speed of a wind farm after two temperatures. ``rank_covariates`` scores each candidate by the absolute
Spearman correlation with the target (the target standardised inside each group, so that panels of series with other levels do not mix their
scales), and marks a covariate as redundant when a higher ranked one is correlated with it above a threshold. A past-only covariate enters the
search as its lag column, so its correlation is the one at the lag of the horizon.
"""

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

#: <covariate_ranking>
REDUNDANCY = 0.95
MIN_ROWS = 20


def _ranks(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def rank_covariates(rows: pd.DataFrame, target: str, candidates: Iterable[str], group_col: Optional[str] = None,
                    redundancy: float = REDUNDANCY) -> pd.DataFrame:
    """The candidates ordered by decreasing score.

    Args:
        rows: Training rows.
        target: Target column.
        candidates: The covariate columns to rank (non-numeric, missing and constant columns get a score of 0).
        group_col: Group column: the target is standardised inside each group.
        redundancy: A covariate whose absolute Spearman correlation with a higher ranked one exceeds this is redundant.

    Returns:
        A frame ``feature, score, redundant_with`` (``redundant_with`` is the higher ranked covariate, else ``None``), best first; ties are
        ordered by name so that the order does not depend on the order of the columns.
    """
    y = pd.to_numeric(rows[target], errors="coerce")
    if group_col and group_col in rows.columns:
        grouped = y.groupby(rows[group_col])
        y = (y - grouped.transform("mean")) / grouped.transform("std").replace(0.0, np.nan)
    scores = {}
    for feature in candidates:
        if feature == target or feature not in rows.columns or not pd.api.types.is_numeric_dtype(rows[feature]):
            scores[feature] = 0.0
            continue
        x = pd.to_numeric(rows[feature], errors="coerce")
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < MIN_ROWS or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            scores[feature] = 0.0
            continue
        rho = _ranks(pair.iloc[:, 0]).corr(_ranks(pair.iloc[:, 1]))
        scores[feature] = float(abs(rho)) if np.isfinite(rho) else 0.0
    order = sorted(scores, key=lambda f: (-scores[f], str(f)))
    redundant_with: dict = {}
    kept: List[str] = []
    for feature in order:
        twin = None
        if scores[feature] > 0.0:
            for other in kept:
                pair = rows[[feature, other]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) >= MIN_ROWS and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1 \
                        and abs(_ranks(pair.iloc[:, 0]).corr(_ranks(pair.iloc[:, 1]))) > redundancy:
                    twin = other
                    break
        redundant_with[feature] = twin
        if twin is None:
            kept.append(feature)
    return pd.DataFrame({"feature": order, "score": [scores[f] for f in order], "redundant_with": [redundant_with[f] for f in order]})
#: </covariate_ranking>
