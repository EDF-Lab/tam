# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The pooled seasonal floor: a seasonal profile that is always in the pool, so the final model is never worse than it on the validation rows.

The floor is a static expert on the position of each row in its natural season (the hour of the day for hourly data, the day of the week for
daily data, the week of the year for weekly data, the month for monthly data, the quarter for quarterly data), one coefficient per position,
fitted in one pass for all the series at once. It needs no feature and no search, so it costs almost nothing, and it is the baseline a
benchmark of seasonal series is compared with.
"""

from typing import Tuple

import numpy as np
import pandas as pd

#: <floor_rules>
FLOOR_ISLAND = "Floor"
FLOOR_COLUMN = "season_pos"


def season_position(dates: pd.Series, steps_per_day: float) -> Tuple[np.ndarray, int]:
    """The position of each date inside its natural season, and the number of positions.

    Args:
        dates: The dates.
        steps_per_day: Steps in one day (24 hourly, 1 daily, 1/7 weekly, 1/30 monthly, 1/91 quarterly).

    Returns:
        ``(positions, count)``: the hour (or step) of the day for a step under a day, the day of the week for daily data, the week of the year
        (0 to 51) for weekly data, the month (0 to 11) for monthly data, the quarter (0 to 3) for quarterly data.

    Raises:
        ValueError: The sampling step has no natural season (yearly data, a step of several days or of a fraction of a day that does not
            divide it).
    """
    d = pd.to_datetime(dates)
    if steps_per_day is None or steps_per_day <= 0:
        raise ValueError("The pooled floor needs a regular sampling step.")
    if steps_per_day > 1.5:
        count = int(round(steps_per_day))
        seconds = (d.dt.hour * 3600 + d.dt.minute * 60 + d.dt.second).to_numpy(dtype=float)
        return np.minimum((seconds / (86400.0 / count)).astype(int), count - 1), count
    if 0.8 <= steps_per_day <= 1.2:
        return d.dt.dayofweek.to_numpy(dtype=int), 7
    if 1 / 9 <= steps_per_day <= 1 / 5:
        return np.minimum((d.dt.dayofyear.to_numpy(dtype=int) - 1) // 7, 51), 52
    if 1 / 35 <= steps_per_day <= 1 / 25:
        return d.dt.month.to_numpy(dtype=int) - 1, 12
    if 1 / 100 <= steps_per_day <= 1 / 85:
        return d.dt.quarter.to_numpy(dtype=int) - 1, 4
    raise ValueError(f"The pooled floor needs a natural season (a step of an hour or less, a day, a week, a month or a quarter): {steps_per_day:g} steps per day has none.")


def floor_formula(target: str, count: int) -> str:
    """Formula of the floor expert: one coefficient per position of the season."""
    return f"{target} ~ c({FLOOR_COLUMN}, n_cat={count}, topo='nominal')"
#: </floor_rules>
