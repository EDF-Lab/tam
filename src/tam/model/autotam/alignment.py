# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Prediction columns aligned to the rows they were asked for.

``StaticTAM.predict`` returns the rows of the frame it is given, sorted and re-indexed when the groups do not share their timestamps (a group
per day of the week of a daily series): the index of the result is then not the index of the input, and a column read by index is missing.
"""

from typing import Optional

import numpy as np
import pandas as pd

#: <alignment>
def aligned_values(predictions: pd.DataFrame, frame: pd.DataFrame, column: str, date_col: Optional[str] = None,
                   group_col: Optional[str] = None) -> np.ndarray:
    """The ``column`` of ``predictions`` in the row order of ``frame``.

    The values are read in place when the date and group columns of the two frames agree row by row. Otherwise rows are matched on those
    columns; without keys, or with other lengths, the index of ``frame`` is used, which gives missing values for the rows that are absent.
    """
    keys = [c for c in (date_col, group_col) if c and c in predictions.columns and c in frame.columns]
    if keys and len(predictions) == len(frame):
        if predictions[keys].reset_index(drop=True).equals(frame[keys].reset_index(drop=True)):
            return predictions[column].to_numpy()               # the same rows in the same order, whatever the index
        left = frame[keys].copy()
        left["__row__"] = np.arange(len(left))
        right = predictions[keys + [column]].drop_duplicates(subset=keys)
        merged = left.merge(right, on=keys, how="left").sort_values("__row__")
        return merged[column].to_numpy()
    if predictions.index.equals(frame.index):
        return predictions[column].to_numpy()
    return predictions[column].reindex(frame.index).to_numpy()
#: </alignment>
