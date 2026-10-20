# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The weighted quantile loss (WQL) of a point forecast with the quantile layer of AutoTAM, as a metric to select experts.

``AutoTAM.predict_quantiles`` returns the point forecast plus the quantiles of its errors. The loss of that layer is the pinball loss of
those quantiles, summed over the levels and divided by the sum of the absolute targets, so it is scale-free like the benchmark's WQL.
Selecting on it instead of the RMSE keeps the experts whose errors are small where the quantile layer needs them (both tails), and is
consistent with the output that is delivered.
"""

from typing import Sequence, Tuple

import numpy as np

#: <quantile_loss>
DEFAULT_LEVELS = (0.1, 0.5, 0.9)


def is_wql(metric: str) -> bool:
    """Whether a metric name is the quantile loss: ``"wql"`` or ``"wql:0.05,0.5,0.95"`` (any case)."""
    return str(metric).lower().startswith("wql")


def parse_levels(metric: str) -> Tuple[float, ...]:
    """The quantile levels of a ``wql`` metric name (``DEFAULT_LEVELS`` for plain ``"wql"``).

    Raises:
        ValueError: A level is not a number in (0, 1), or the list is empty.
    """
    text = str(metric).lower()
    if ":" not in text:
        return DEFAULT_LEVELS
    try:
        levels = tuple(float(part) for part in text.split(":", 1)[1].split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"The levels of {metric!r} must be numbers, for example 'wql:0.1,0.5,0.9'.") from exc
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(f"The levels of {metric!r} must lie in (0, 1), for example 'wql:0.1,0.5,0.9'.")
    return tuple(sorted(levels))


def wql(y_true: np.ndarray, y_pred: np.ndarray, levels: Sequence[float] = DEFAULT_LEVELS) -> float:
    """Weighted quantile loss of ``y_pred`` with the quantiles of its own errors as the quantile layer.

    Args:
        y_true: Observed values (finite).
        y_pred: Point forecasts (finite), same length.
        levels: Quantile levels in (0, 1).

    Returns:
        ``mean over levels of 2 * sum(pinball loss) / sum(|y_true|)``; ``inf`` when there is nothing to score.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("inf")
    errors = y_true - y_pred
    total = 0.0
    for level, offset in zip(levels, np.quantile(errors, list(levels))):
        shortfall = y_true - (y_pred + offset)
        total += float(np.sum(np.maximum(level * shortfall, (level - 1.0) * shortfall)))
    scale = float(np.sum(np.abs(y_true)))
    return 2.0 * total / (len(levels) * (scale if scale > 0 else 1e-9))
#: </quantile_loss>
