# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Conformal calibration of a distributional StaticTAM.

A distributional (dict-formula) StaticTAM gives a sharp, heteroscedastic shape, but its coverage is only as
good as the assumed tail law F. This wraps it with finite-sample, distribution-free validity regardless of F:
conformalized quantile-regression (CQR) intervals, a conformal p-value anomaly score, Mondrian (stratified)
calibration, and Adaptive Conformal Inference for streaming coverage under drift.

It duck-types the model, needing only predict_quantile / anomaly_score / _mu_sigma / _to_model_scale /
target_col_ / _log_target_ - the surface any distributional StaticTAM exposes. The conformal mechanics (the
finite-sample quantile, the p-value, the ACI level update) are delegated to the static SafetyTAM and the
streaming aci module; this class only computes the model-specific nonconformity scores (a studentized
residual and the CQR score) and assembles the outputs. One source of truth for conformal.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...safety import SafetyTAM
from .aci import update_risk_level, effective_alpha

_ALL_STRATA = "__all__"


class ConformalDistributionalTAM:
    """Wrap a fitted distributional StaticTAM with split-conformal calibration, p-values and ACI."""

    def __init__(self, distributional_model, alpha: float = 0.1, strata_col: Optional[str] = None):
        """distributional_model is a fitted dict-formula StaticTAM; alpha the target miscoverage; strata_col
        an optional column for Mondrian (per-stratum) calibration."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.model = distributional_model
        self.alpha = float(alpha)
        self.strata_col = strata_col
        self._studentized_engine: Dict[object, SafetyTAM] = {}
        self._cqr_engine: Dict[object, SafetyTAM] = {}

    def _strata(self, data: pd.DataFrame) -> np.ndarray:
        if self.strata_col is None:
            return np.full(len(data), _ALL_STRATA, dtype=object)
        return data[self.strata_col].to_numpy()

    def _studentized_scores(self, data: pd.DataFrame) -> np.ndarray:
        return np.abs(self.model.anomaly_score(data)["z_score"].to_numpy())

    def _engine(self, table: Dict[object, SafetyTAM], stratum: object) -> SafetyTAM:
        return table.get(stratum, table[_ALL_STRATA])

    def calibrate(self, calibration_data: pd.DataFrame) -> "ConformalDistributionalTAM":
        """Fit one SafetyTAM per stratum on the CQR and studentized nonconformity scores."""
        lower_quantile = self.model.predict_quantile(calibration_data, self.alpha / 2.0)
        upper_quantile = self.model.predict_quantile(calibration_data, 1.0 - self.alpha / 2.0)
        observed = calibration_data[self.model.target_col_].to_numpy()
        cqr_scores = np.maximum(lower_quantile - observed, observed - upper_quantile)
        studentized_scores = self._studentized_scores(calibration_data)
        strata = self._strata(calibration_data)

        for stratum in list(np.unique(strata)) + [_ALL_STRATA]:
            mask = np.ones(len(strata), dtype=bool) if stratum is _ALL_STRATA else (strata == stratum)
            self._studentized_engine[stratum] = SafetyTAM(self.alpha).calibrate_scores(studentized_scores[mask])
            self._cqr_engine[stratum] = SafetyTAM(self.alpha).calibrate_scores(cqr_scores[mask])
        return self

    def predict_interval(self, data: pd.DataFrame) -> pd.DataFrame:
        """CQR interval with finite-sample 1 - alpha coverage (per-stratum widths when Mondrian)."""
        lower_quantile = self.model.predict_quantile(data, self.alpha / 2.0)
        upper_quantile = self.model.predict_quantile(data, 1.0 - self.alpha / 2.0)
        widths = np.array([self._engine(self._cqr_engine, s).conformal_quantile(self.alpha)
                           for s in self._strata(data)])
        return pd.DataFrame({"lower": lower_quantile - widths, "upper": upper_quantile + widths}, index=data.index)

    def conformal_pvalue(self, data: pd.DataFrame) -> np.ndarray:
        """Distribution-free conformal p-value of each observation (small => anomalous)."""
        scores = self._studentized_scores(data)
        strata = self._strata(data)
        pvalues = np.empty(len(data), dtype=float)
        for stratum in np.unique(strata):
            mask = strata == stratum
            pvalues[mask] = self._engine(self._studentized_engine, stratum).pvalue(scores[mask])
        return pvalues

    def anomaly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Conformal p-value, the anomaly flag at level alpha, and the calibrated interval."""
        pvalue = self.conformal_pvalue(data)
        interval = self.predict_interval(data)
        return pd.DataFrame({
            "conformal_pvalue": pvalue,
            "is_anomaly": pvalue < self.alpha,
            "lower": interval["lower"].to_numpy(),
            "upper": interval["upper"].to_numpy(),
        }, index=data.index)

    def aci_intervals(self, ordered_data: pd.DataFrame, gamma: float = 0.02) -> pd.DataFrame:
        """Adaptive Conformal Inference: intervals whose level adapts online to hold coverage under drift.

        A single global risk level alpha_t adapts row by row (via aci.update_risk_level), while the conformal
        radius is drawn from the row's stratum engine (SafetyTAM.conformal_quantile). This is the
        Mondrian-stratified, location-scale specialisation of the generic ACI loop.
        """
        mu_hat, sigma_hat = self.model._mu_sigma(ordered_data)
        observed = self.model._to_model_scale(ordered_data[self.model.target_col_].to_numpy())
        strata = self._strata(ordered_data)
        alpha_t = self.alpha
        lowers, uppers, levels = [], [], []
        for position in range(len(ordered_data)):
            radius = self._engine(self._studentized_engine, strata[position]).conformal_quantile(
                effective_alpha(alpha_t)
            )
            lower = mu_hat[position] - sigma_hat[position] * radius
            upper = mu_hat[position] + sigma_hat[position] * radius
            inside = lower <= observed[position] <= upper
            lowers.append(np.exp(lower) if self.model._log_target_ else lower)
            uppers.append(np.exp(upper) if self.model._log_target_ else upper)
            levels.append(alpha_t)
            alpha_t = update_risk_level(alpha_t, 0.0 if inside else 1.0, self.alpha, gamma)
        return pd.DataFrame({"lower": lowers, "upper": uppers, "alpha_t": levels}, index=ordered_data.index)
