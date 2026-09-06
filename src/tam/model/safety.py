# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Safety module: the framework's static, distribution-free conformal engine.

Split (i.i.d.) conformal prediction that guarantees marginal coverage regardless of the model. It is the
single conformal core the risk layer builds on: the CQR / normalized wrapper
(tam.model.statistics.risk.conformal.ConformalDistributionalTAM) delegates its finite-sample quantile and
p-value here. The streaming, non-stationary counterpart (Adaptive Conformal Inference) lives separately in
tam.model.statistics.risk.aci, keeping drift adaptation apart from static calibration.

Two layers: generic primitives on a stream of nonconformity scores (calibrate_scores, conformal_quantile,
pvalue), and a symmetric-interval convenience (calibrate, predict_intervals).
"""

import numpy as np
import pandas as pd
from typing import Optional

_TINY: float = 1e-12


class SafetyTAM:
    """Static split-conformal prediction: a finite-sample marginal coverage guarantee under exchangeability.

    For coverage under distribution shift, feed this engine's conformal_quantile to the Adaptive Conformal
    Inference loop in tam.model.statistics.risk.aci.
    """

    def __init__(self, alpha: float = 0.1):
        """alpha is the target error rate (e.g. 0.1 for 90% coverage)."""
        self.alpha_target = alpha
        self.residuals_calib_ = None

    def calibrate_scores(self, scores: np.ndarray) -> "SafetyTAM":
        """Calibrate directly on a supplied array of nonconformity scores (the generic entry point)."""
        self.residuals_calib_ = np.asarray(scores, dtype=float)
        return self

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scale: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
    ) -> "SafetyTAM":
        """Calibrate on a hold-out set.

        The nonconformity score defaults to the absolute residual `|y - y_pred|`; passing scale normalizes it
        to `|y - y_pred|` / scale (studentized conformal), and passing scores uses them directly.
        """
        if scores is not None:
            self.residuals_calib_ = np.asarray(scores, dtype=float)
        else:
            residual = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
            self.residuals_calib_ = residual / np.asarray(scale, dtype=float) if scale is not None else residual
        print(f"Safety calibrated on {len(self.residuals_calib_)} samples.")
        return self

    def conformal_quantile(self, alpha: Optional[float] = None) -> float:
        """Finite-sample-corrected (1 - alpha) empirical quantile of the calibration scores."""
        current_alpha = self.alpha_target if alpha is None else alpha
        n = len(self.residuals_calib_)
        q_level = np.clip((1 - current_alpha) * (1 + 1 / n), 0, 1)
        return float(np.quantile(self.residuals_calib_, q_level))

    def _get_quantile(self, current_alpha: float) -> float:
        """Backward-compatible alias for conformal_quantile."""
        return self.conformal_quantile(current_alpha)

    def pvalue(self, scores: np.ndarray) -> np.ndarray:
        """Distribution-free conformal p-values (1 + #{cal >= s}) / (n + 1) for new scores."""
        if self.residuals_calib_ is None:
            raise RuntimeError("You must call .calibrate() or .calibrate_scores() before scoring p-values.")
        sorted_calibration = np.sort(self.residuals_calib_)
        n = len(sorted_calibration)
        query = np.atleast_1d(np.asarray(scores, dtype=float))
        greater_equal = n - np.searchsorted(sorted_calibration, query, side="left")
        return (1.0 + greater_equal) / (n + 1.0)

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        y_true_online: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Static split-conformal intervals [Lower, Upper] = y_pred +/- q * scale.

        A single fixed radius q = conformal_quantile(alpha_target) is used throughout (constant-width, or
        heteroscedastic when scale is supplied). For coverage that adapts online under drift, use
        tam.model.statistics.risk.aci.adaptive_conformal_intervals with this engine's conformal_quantile.

        Returns a DataFrame with ['Predicted', 'Lower', 'Upper', 'Alpha_t', 'Width'] (+ 'Actual', 'Covered').
        """
        if self.residuals_calib_ is None:
            raise RuntimeError("You must call .calibrate() before predicting intervals.")

        y_pred = np.asarray(y_pred, dtype=float)
        n_test = len(y_pred)
        scale_vector = np.ones(n_test) if scale is None else np.asarray(scale, dtype=float)

        static_radius = self.conformal_quantile(self.alpha_target)
        radii = np.full(n_test, static_radius)
        alphas = np.full(n_test, self.alpha_target)

        half_widths = radii * scale_vector
        result = pd.DataFrame({
            'Predicted': y_pred,
            'Lower': y_pred - half_widths,
            'Upper': y_pred + half_widths,
            'Alpha_t': alphas,
            'Width': 2 * half_widths,
        })
        if y_true_online is not None:
            result['Actual'] = np.asarray(y_true_online, dtype=float)
            result['Covered'] = (result['Actual'] >= result['Lower']) & (result['Actual'] <= result['Upper'])
        return result
