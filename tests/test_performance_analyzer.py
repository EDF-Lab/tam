# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.evaluation.performance_analyzer``, residual diagnostics
and temporal-degradation detection.
"""

import numpy as np

from tam.evaluation.performance_analyzer import analyze_residuals, detect_temporal_degradation


def test_analyze_residuals_reports_expected_keys():
    rng = np.random.default_rng(0)
    y_true = rng.normal(50, 5, 100)
    y_pred = y_true + rng.normal(0, 1, 100)
    stats = analyze_residuals(y_true, y_pred)

    assert set(stats) == {"Mean Error (Bias)", "Std Error", "Skewness", "Kurtosis", "Lag-1 AutoCorr"}
    assert np.isfinite(stats["Mean Error (Bias)"])
    assert np.isfinite(stats["Lag-1 AutoCorr"])


def test_analyze_residuals_all_nan_returns_empty():
    y_true = np.array([np.nan, np.nan, np.nan])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert analyze_residuals(y_true, y_pred) == {}


def test_analyze_residuals_single_point_has_nan_autocorr():
    stats = analyze_residuals(np.array([10.0]), np.array([9.0]))
    assert np.isnan(stats["Lag-1 AutoCorr"])


def test_detect_degradation_flags_growing_error():
    # First half tracks closely; second half drifts badly -> positive degradation.
    y_true = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    y_pred = np.array([10.1, 11.1, 11.9, 13.1, 20.0, 25.0, 8.0, 30.0])
    degradation = detect_temporal_degradation(y_true, y_pred, metric="RMSE")
    assert degradation > 0.0


def test_detect_degradation_too_few_samples_returns_nan():
    assert np.isnan(detect_temporal_degradation(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_detect_degradation_perfect_first_half_returns_nan():
    """A zero first-half error makes the ratio undefined."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.0, 2.0, 3.0, 9.0, 8.0, 7.0])  # first half exact -> h1 RMSE = 0
    assert np.isnan(detect_temporal_degradation(y_true, y_pred, metric="RMSE"))
