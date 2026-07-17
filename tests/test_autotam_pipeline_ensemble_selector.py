# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.pipeline.ensemble_selector.EnsembleSelector``.

The full evaluate_and_refit() orchestration is not exercised here because it
requires a complete pipeline run. We test the metric calculator, which is the
only pure-math component in this module.
"""

import numpy as np
import pytest
from tam.model.autotam.pipeline.ensemble_selector import EnsembleSelector


def _sel() -> EnsembleSelector:
    return EnsembleSelector()


def test_calculate_error_rmse():
    sel = _sel()
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    rmse = sel._calculate_error(y_true, y_pred, "rmse")
    assert np.isclose(rmse, 1.0)


def test_calculate_error_mae():
    sel = _sel()
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    assert np.isclose(sel._calculate_error(y_true, y_pred, "mae"), 1.0)


def test_calculate_error_mape():
    sel = _sel()
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # MAPE: (10/100 + 20/200) / 2 * 100 = 10%
    assert np.isclose(sel._calculate_error(y_true, y_pred, "mape"), 10.0)


def test_calculate_error_unknown_metric_falls_back_to_rmse():
    sel = _sel()
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 19.0])
    result = sel._calculate_error(y_true, y_pred, "unknown_metric")
    assert np.isfinite(result)


def test_calculate_error_nan_mask():
    sel = _sel()
    y_true = np.array([10.0, np.nan, 30.0])
    y_pred = np.array([11.0, 99.0, 31.0])
    # The NaN row is masked; only the two clean rows contribute.
    rmse = sel._calculate_error(y_true, y_pred, "rmse")
    assert np.isclose(rmse, 1.0)


def test_calculate_error_all_nan_returns_inf():
    sel = _sel()
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([1.0, 2.0])
    assert sel._calculate_error(y_true, y_pred, "rmse") == float("inf")
