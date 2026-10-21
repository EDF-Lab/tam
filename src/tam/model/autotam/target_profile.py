# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Facts about the target that make a fit wrong when nobody looks.

A station's missing-value code fitted as a value, a counter that only grows, a count or a proportion modelled as a Gaussian: ``profile_target``
reads these facts from the training target and gives a family hint; ``mask_sentinels`` turns a sentinel into a missing value.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

#: <target_profile_rules>
SENTINEL_MIN_COUNT = 2
SENTINEL_MIN_SHARE = 0.001
SENTINEL_ROBUST_SIGMAS = 20.0
MONOTONE_SHARE = 0.99
INTEGER_SHARE = 0.99
ZERO_MASS_SHARE = 0.10
SKEW_LIMIT = 1.0
KURTOSIS_LIMIT = 5.0


def _sentinels(values: np.ndarray) -> List[float]:
    """Values that repeat at an extreme of the range, far outside the robust spread: a missing-value code such as -9999."""
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if scale == 0.0:
        scale = float(np.std(values)) or 1.0
    far = np.abs(values - center) > SENTINEL_ROBUST_SIGMAS * scale
    if not far.any():
        return []
    needed = max(SENTINEL_MIN_COUNT, int(np.ceil(SENTINEL_MIN_SHARE * values.size)))
    low, high = float(values.min()), float(values.max())
    found = []
    for value, count in pd.Series(values[far]).value_counts().items():
        if count >= needed and value in (low, high):
            found.append(float(value))
    return sorted(found)


def profile_target(y: pd.Series) -> Dict[str, Any]:
    """Profile of a training target: sentinels, bounds, monotone and integer shares, shape, and a family hint.

    Args:
        y: Training target, missing values allowed.

    Returns:
        A dictionary: ``n``, ``sentinels`` (list), ``bounds`` (``(low, high)`` or ``None``), ``monotone_share``, ``integer_share``, ``zero_share``,
        ``negative_share``, ``skewness``, ``kurtosis`` and ``hints``, a one-item list among ``cumulative``, ``bounded``, ``count``, ``tweedie``,
        ``log``, ``asinh``, ``binomial`` (a target that only takes 0 and 1) and ``gaussian`` (the family hints of the model of the target).
    """
    values = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    profile: Dict[str, Any] = {"n": int(values.size), "sentinels": [], "bounds": None, "monotone_share": 0.0, "integer_share": 0.0,
                               "zero_share": 0.0, "negative_share": 0.0, "skewness": 0.0, "kurtosis": 0.0, "hints": ["gaussian"]}
    if values.size < 5:
        return profile
    sentinels = _sentinels(values)
    profile["sentinels"] = sentinels
    if sentinels:
        values = values[~np.isin(values, sentinels)]
    steps = np.diff(values)
    monotone = max(float((steps >= 0).mean()), float((steps <= 0).mean())) if steps.size and np.any(steps != 0) else 0.0
    series = pd.Series(values)
    low, high = float(values.min()), float(values.max())
    integer_share = float(np.mean(values == np.round(values)))
    profile.update(monotone_share=monotone, integer_share=integer_share, zero_share=float(np.mean(values == 0.0)),
                   negative_share=float(np.mean(values < 0.0)), skewness=float(series.skew()), kurtosis=float(series.kurt()))
    if low >= 0.0 and high <= 1.0 and integer_share < INTEGER_SHARE:
        profile["bounds"] = (0.0, 1.0)
    elif low >= 0.0 and high == 100.0:
        profile["bounds"] = (0.0, 100.0)

    binary = bool(np.isin(values, (0.0, 1.0)).all()) and np.unique(values).size <= 2
    if binary:
        profile["bounds"] = (0.0, 1.0)
    if monotone >= MONOTONE_SHARE and not binary:
        hint = "cumulative"
    elif binary:
        hint = "binomial"          # a 0/1 target is a Bernoulli outcome, not a count: a log link would forecast above one
    elif profile["bounds"] is not None:
        hint = "bounded"
    elif integer_share >= INTEGER_SHARE and low >= 0.0:
        hint = "count"
    elif low >= 0.0 and profile["zero_share"] >= ZERO_MASS_SHARE:
        hint = "tweedie"
    elif low > 0.0 and profile["skewness"] > SKEW_LIMIT:
        hint = "log"
    elif low < 0.0 and (abs(profile["skewness"]) > SKEW_LIMIT or profile["kurtosis"] > KURTOSIS_LIMIT):
        hint = "asinh"
    else:
        hint = "gaussian"
    profile["hints"] = [hint]
    return profile


def mask_sentinels(df: pd.DataFrame, target: str, sentinels: List[float]) -> pd.DataFrame:
    """Copy of ``df`` where the sentinel values of the target are missing values (the input is left alone)."""
    out = df.copy()
    if sentinels and target in out.columns:
        out.loc[out[target].isin(sentinels), target] = np.nan
    return out
#: </target_profile_rules>
