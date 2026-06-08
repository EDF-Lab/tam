r"""
Unit tests for ``tam.evaluation.metrics.calculate_regression_metrics``.

Covers the standard error metrics plus the NaN/Inf masking and zero-division
edge cases that the evolutionary search relies on for robustness.
"""

import numpy as np

from tam.evaluation.metrics import calculate_regression_metrics


def test_metrics_on_clean_data():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])
    m = calculate_regression_metrics(y_true, y_pred)

    assert set(m) >= {"RMSE", "MAE", "NMAE", "SMAPE", "R2", "MAPE"}
    assert np.isclose(m["MAE"], 1.0)
    assert np.isclose(m["RMSE"], 1.0)
    assert m["R2"] > 0.99


def test_metrics_empty_prediction_returns_empty():
    assert calculate_regression_metrics(np.array([]), np.array([])) == {}


def test_metrics_all_nan_returns_empty():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([1.0, 2.0])
    assert calculate_regression_metrics(y_true, y_pred) == {}


def test_metrics_mask_drops_nan_and_inf():
    y_true = np.array([10.0, np.nan, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, np.inf, 39.0])
    m = calculate_regression_metrics(y_true, y_pred)
    # Only the two fully-finite pairs survive the mask.
    assert np.isclose(m["MAE"], 1.0)


def test_mape_is_nan_when_target_contains_zero():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 11.0, 19.0])
    m = calculate_regression_metrics(y_true, y_pred)
    assert np.isnan(m["MAPE"])


def test_mape_is_finite_without_zeros():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    m = calculate_regression_metrics(y_true, y_pred)
    assert np.isfinite(m["MAPE"])


def test_r2_is_nan_for_constant_target():
    """Zero target variance makes R2 undefined."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    m = calculate_regression_metrics(y_true, y_pred)
    assert np.isnan(m["R2"])
