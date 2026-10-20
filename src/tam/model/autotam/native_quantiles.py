# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Experts that forecast quantiles themselves.

The quantile layer of ``AutoTAM.predict_quantiles`` is, by default, the point forecast of the final model plus the quantiles of its
validation errors: every expert has the same error shape. A distributional ``StaticTAM`` models the location and the scale of the target
from the features, so its quantiles widen where the data are noisier. ``NativeQuantileExpert`` wraps one so that it joins the pool like
any other static expert (its point forecast is the median) and keeps its quantiles for ``predict_quantiles(aggregation="opera")``.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd

from tam.common.utils import parse_formula_to_terms
from tam.model.additive import StaticTAM

#: <native_quantiles>
DISTRIBUTION = {"log_target": False, "tail_family": "normal"}   # the target on its own scale, a normal tail


def scale_formula(target: str, location_formula: str) -> Optional[str]:
    """The formula of the scale: linear in the numeric features of the location formula, ``None`` when it has none."""
    _, terms = parse_formula_to_terms(location_formula)
    features = []
    for term in terms:
        feature = term.get("feature")
        if term.get("type") in ("c", "te") or not feature or feature == "interaction" or feature in features:
            continue
        features.append(feature)
    return f"{target} ~ " + " + ".join(f"l({f})" for f in features) if features else None


class NativeQuantileExpert:
    """A distributional ``StaticTAM`` (location from the champion formula, scale linear in its features) that behaves as a static expert."""

    def __init__(self, target: str, location_formula: str, group_col: Optional[str] = None, date_col: Optional[str] = None):
        scale = scale_formula(target, location_formula)
        if scale is None:
            raise ValueError("The location formula has no numeric feature to model the scale on.")
        self.target_col_ = target
        self.formula_ = location_formula          # the location formula: what the callers read the columns of the expert from
        self.scale_formula_ = scale
        self.group_col, self.date_col = group_col, date_col
        self.model_: Optional[StaticTAM] = None

    def fit(self, data: pd.DataFrame) -> "NativeQuantileExpert":
        model = StaticTAM({"mu": self.formula_, "sigma": self.scale_formula_}, loss={"mu": "l2", "sigma": "gamma"},
                          group_col=self.group_col, date_col=self.date_col, dist_kwargs=dict(DISTRIBUTION))
        self.model_ = model.fit(data)
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """The median, in the column ``Estimated<target>`` of a static expert."""
        median = self.model_.predict_median(data)
        return pd.DataFrame({f"Estimated{self.target_col_}": pd.Series(median).to_numpy(dtype=float)}, index=data.index)

    def predict_quantiles(self, data: pd.DataFrame, taus: Sequence[float]) -> pd.DataFrame:
        """One ``q<tau>`` column per level, sorted, on the rows of ``data``."""
        quantiles = self.model_.predict_quantiles(data, taus=list(taus))
        quantiles.index = data.index
        return quantiles
#: </native_quantiles>
