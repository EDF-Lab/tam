# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Effects of a static model summed by kind, as inputs of the adaptive and Kalman stages.

An online stage that tracks one coefficient per effect has as many parameters as the base model has effects, and little data per window.
Summing the effects by kind (the exogenous drivers, the calendar, the autoregressive terms) leaves a handful of inputs whose coefficients
rescale each block, which is the grouped form of the benchmark protocol for load forecasting. ``GroupedEffectsTAM`` is the base model with
one more column per kind in its decomposition, so ``AdaptiveTAM`` and ``KalmanTAM`` take them as ordinary inputs.
"""

import copy
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from tam.model.additive import StaticTAM

#: <effect_groups>
GROUP_PREFIX = "group_effect_"
GROUP_KINDS = ("exogenous", "calendar", "autoregressive")
CALENDAR_FEATURES = frozenset({
    "toy", "tod", "dow", "hour", "weekday", "dayofweek", "day_of_week", "month", "dayofyear", "day_of_year", "week", "quarter",
    "holiday", "days_to_holiday", "days_since_holiday", "bridge_day",
})
HOLIDAY_FLAGS = ("holiday", "bridge_day")


def calendar_flags(columns: Iterable[str]) -> List[str]:
    """The 0/1 calendar columns an adaptive stage may switch on: the holiday flags and the ``crisis_<name>`` indicators (not their ages)."""
    names = list(columns)
    return [c for c in HOLIDAY_FLAGS if c in names] + [c for c in names if str(c).startswith("crisis_") and not str(c).endswith("_age")]


def effect_kind(feature: str, calendar: Iterable[str] = ()) -> str:
    """The kind of an effect: ``autoregressive`` (a lag of the target or of a covariate), ``calendar``, else ``exogenous``.

    Args:
        feature: Name of the feature of the effect, without the ``effect_`` prefix (an interaction names its features).
        calendar: Extra column names that are calendar columns (the ones added by ``calendar_features``).
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", feature)
    if any("_lag_" in token for token in tokens):
        return "autoregressive"
    known = CALENDAR_FEATURES | set(calendar)
    return "calendar" if any(token in known for token in tokens) else "exogenous"


class GroupedEffectsTAM(StaticTAM):
    """A fitted ``StaticTAM`` whose decomposition also holds ``group_effect_<kind>``: the sum of the effects of each kind."""

    _calendar_columns: List[str] = []

    def decompose_prediction(self, *args, **kwargs):
        out = super().decompose_prediction(*args, **kwargs)
        sums: Dict[str, List[str]] = {}
        for column in out.columns:
            if str(column).startswith("effect_") and column != "effect_offset":   # the intercept is carried by the base forecast
                sums.setdefault(effect_kind(str(column)[len("effect_"):], self._calendar_columns), []).append(column)
        for kind in GROUP_KINDS:
            if kind in sums:
                out[GROUP_PREFIX + kind] = out[sums[kind]].sum(axis=1)
        return out


def with_effect_groups(model: StaticTAM, calendar: Iterable[str] = ()) -> Optional[GroupedEffectsTAM]:
    """A copy of a fitted plain ``StaticTAM`` that decomposes into grouped effects too (``None`` for any other model)."""
    if type(model) is not StaticTAM or getattr(model, "coefficients_", None) is None:
        return None
    grouped = copy.copy(model)
    grouped.__class__ = GroupedEffectsTAM
    grouped._calendar_columns = list(calendar)
    return grouped


def group_columns(decomposition: pd.DataFrame) -> List[str]:
    """The grouped columns of a decomposition, in the order of ``GROUP_KINDS``."""
    return [GROUP_PREFIX + kind for kind in GROUP_KINDS if GROUP_PREFIX + kind in decomposition.columns]
#: </effect_groups>
