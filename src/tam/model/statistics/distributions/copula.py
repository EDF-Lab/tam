# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Gaussian copula over several distributional StaticTAM margins.

Sklar's theorem separates a joint model into independent marginals plus a copula for the dependence.
Fit each margin with a distributional (dict-formula) StaticTAM, map each through its probability integral
transform to a normal score, then estimate the score correlation R. Beyond joint density this gives a
joint anomaly score (Mahalanobis distance of the normal scores under R), flagging observations that are
jointly improbable even when each margin looks individually ordinary.

This is the one distributional object NOT folded into StaticTAM: it binds several fitted marginal models,
so it stays a standalone class. It only needs each margin's .cdf() and .fit(), never imports StaticTAM.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

_TINY: float = 1e-9


class GaussianCopulaTAM:
    """Bind distributional StaticTAM margins (fitted or unfitted) with a Gaussian copula."""

    def __init__(self, margins: Dict[str, "object"]):
        """margins maps a name to a distributional (dict-formula) StaticTAM (one per response)."""
        if len(margins) < 2:
            raise ValueError("A copula requires at least two margins.")
        self.margins = margins
        self.margin_names_ = list(margins.keys())
        self.correlation_: np.ndarray = np.eye(len(margins))
        self._correlation_inverse: np.ndarray = np.eye(len(margins))
        self._log_determinant: float = 0.0

    def _normal_scores_from_uniform(self, data: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.margin_names_:
            uniform = np.clip(self.margins[name].cdf(data), _TINY, 1.0 - _TINY)
            columns.append(stats.norm.ppf(uniform))
        return np.stack(columns, axis=1)

    def fit(self, data: pd.DataFrame) -> "GaussianCopulaTAM":
        """Fit every margin (if not already), then estimate the copula correlation from normal scores."""
        for name in self.margin_names_:
            if getattr(self.margins[name], "_location_submodel_", None) is None:
                self.margins[name].fit(data)
        normal_scores = self._normal_scores_from_uniform(data)
        self.correlation_ = np.corrcoef(normal_scores, rowvar=False)
        # A ridge jitter keeps the empirical correlation matrix invertible when margins are near-collinear.
        regularized_correlation = self.correlation_ + 1e-6 * np.eye(len(self.margin_names_))
        self._correlation_inverse = np.linalg.inv(regularized_correlation)
        sign, log_determinant = np.linalg.slogdet(regularized_correlation)
        self._log_determinant = float(log_determinant)
        return self

    def normal_scores(self, data: pd.DataFrame) -> np.ndarray:
        """The per-margin normal scores Phi^-1(F_j(y_j)). Shape (n, d)."""
        return self._normal_scores_from_uniform(data)

    def copula_log_density(self, data: pd.DataFrame) -> np.ndarray:
        """Log density of the Gaussian copula (the dependence contribution) at each observation."""
        scores = self._normal_scores_from_uniform(data)
        quadratic = np.einsum("nd,de,ne->n", scores, self._correlation_inverse - np.eye(len(self.margin_names_)), scores)
        return -0.5 * (self._log_determinant + quadratic)

    def joint_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Joint abnormality: Mahalanobis distance of the normal scores, its chi-square p-value and surprisal."""
        scores = self._normal_scores_from_uniform(data)
        mahalanobis = np.einsum("nd,de,ne->n", scores, self._correlation_inverse, scores)
        degrees_of_freedom = len(self.margin_names_)
        joint_pvalue = np.clip(stats.chi2.sf(mahalanobis, degrees_of_freedom), _TINY, 1.0)
        return pd.DataFrame({
            "mahalanobis": mahalanobis,
            "joint_pvalue": joint_pvalue,
            "joint_anomaly_score": -np.log(joint_pvalue),
        }, index=data.index)
