# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Extreme Value Theory for rigorous tail / severe-anomaly scoring.

A parametric location-scale fit constrains the whole shape and may misfit the extreme tail - exactly where
severe anomalies live. By the Pickands-Balkema-de Haan theorem, exceedances of a high threshold u converge
to a Generalized Pareto Distribution, giving asymptotically-justified tail probabilities and return levels a
bulk family cannot guarantee. fit_gpd_tail extracts the standardized-residual magnitude from a fitted
distributional StaticTAM and models its upper tail, so extreme observations get a principled EVT score.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto

_TINY: float = 1e-300


class GeneralizedParetoTail:
    """A peaks-over-threshold GPD model for the upper tail of a nonnegative magnitude sample."""

    def __init__(self, threshold_quantile: float = 0.95, max_shape: float = 1.0):
        """threshold_quantile is the high threshold u (an empirical quantile); max_shape caps the fitted GPD
        shape xi (GPD-MLE is sensitive to the top order statistics and xi >= 1 means an infinite-mean tail,
        so the shape is clamped and a warning is emitted when the raw estimate reaches the cap)."""
        if not 0.5 <= threshold_quantile < 1.0:
            raise ValueError(f"threshold_quantile must be in [0.5, 1); got {threshold_quantile}")
        self.threshold_quantile = float(threshold_quantile)
        self.max_shape = float(max_shape)
        self.threshold_: float = 0.0
        self.shape_: float = 0.0
        self.scale_: float = 1.0
        self.exceedance_rate_: float = 0.0

    def fit(self, magnitudes: np.ndarray) -> "GeneralizedParetoTail":
        """Fit the GPD (shape, scale) to threshold exceedances by maximum likelihood."""
        values = np.asarray(magnitudes, dtype=float)
        self.threshold_ = float(np.quantile(values, self.threshold_quantile))
        exceedances = values[values > self.threshold_] - self.threshold_
        if exceedances.size < 10:
            raise ValueError("Too few threshold exceedances to fit a GPD; lower threshold_quantile.")
        fitted_shape, _, self.scale_ = genpareto.fit(exceedances, floc=0.0)
        if fitted_shape >= self.max_shape:
            warnings.warn(
                f"GPD shape xi={fitted_shape:.3f} reached the cap {self.max_shape} (an infinite-mean tail). "
                "Clamping for stable tail scores - inspect for a dominating outlier or raise threshold_quantile.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.shape_ = float(min(fitted_shape, self.max_shape))
        self.exceedance_rate_ = float(np.mean(values > self.threshold_))
        return self

    def tail_probability(self, values: np.ndarray) -> np.ndarray:
        """P(M > value) from the fitted GPD above the threshold; 1 below it (not extreme)."""
        excess = np.asarray(values, dtype=float) - self.threshold_
        survival = genpareto.sf(np.clip(excess, 0.0, None), self.shape_, loc=0.0, scale=self.scale_)
        return np.where(excess > 0.0, self.exceedance_rate_ * survival, 1.0)

    def return_level(self, exceedance_probability: float) -> float:
        """The magnitude exceeded with probability exceedance_probability (must be below the rate)."""
        ratio = exceedance_probability / self.exceedance_rate_
        if abs(self.shape_) < 1e-6:
            return self.threshold_ - self.scale_ * np.log(ratio)
        return self.threshold_ + self.scale_ / self.shape_ * (ratio ** (-self.shape_) - 1.0)

    def anomaly_score(self, values: np.ndarray) -> np.ndarray:
        """EVT surprisal -log P(M > value) - large for extreme observations."""
        return -np.log(np.clip(self.tail_probability(values), _TINY, 1.0))


def fit_gpd_tail(distributional_model, data: pd.DataFrame, threshold_quantile: float = 0.95) -> GeneralizedParetoTail:
    """Fit a GeneralizedParetoTail to the standardized-residual magnitude of a distributional StaticTAM."""
    magnitudes = np.abs(distributional_model.anomaly_score(data)["z_score"].to_numpy())
    return GeneralizedParetoTail(threshold_quantile=threshold_quantile).fit(magnitudes)
