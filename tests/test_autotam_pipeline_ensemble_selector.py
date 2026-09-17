# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.pipeline.ensemble_selector.EnsembleSelector``.

The full evaluate_and_refit() orchestration is not exercised here because it
requires a complete pipeline run. We test the pure components: the metric
calculator and the Apex quality filter.
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


def _experts(scores):
    return [{"name": f"e{i}", "val_rmse": score} for i, score in enumerate(scores)]


def test_apex_quality_filter_drops_experts_beyond_the_ratio():
    sel = EnsembleSelector(apex_quality_ratio=2.0)
    kept = sel._filter_apex_pool(_experts([1000.0, 1900.0, 2000.0, 2100.0, 9700.0]))
    assert [e["name"] for e in kept] == ["e0", "e1", "e2"]


def test_apex_quality_filter_defaults_to_one_and_a_half():
    """The shipped floor. On the national benchmark 1.5x beat 2x on 2 of 3 seeds and tied on the third."""
    assert EnsembleSelector().apex_quality_ratio == 1.5


def test_apex_quality_filter_disabled_keeps_the_pool():
    sel = EnsembleSelector(apex_quality_ratio=None)
    pool = _experts([1000.0, 9700.0, 17000.0])
    assert sel._filter_apex_pool(pool) == pool


def test_apex_quality_filter_keeps_the_two_best_when_only_one_passes():
    sel = EnsembleSelector(apex_quality_ratio=1.5)
    kept = sel._filter_apex_pool(_experts([5000.0, 1000.0, 9000.0, 4000.0]))
    assert [e["name"] for e in kept] == ["e1", "e3"]


def test_apex_quality_filter_drops_non_finite_scores():
    sel = EnsembleSelector(apex_quality_ratio=2.0)
    kept = sel._filter_apex_pool(_experts([1000.0, np.inf, np.nan, 1500.0]))
    assert [e["name"] for e in kept] == ["e0", "e3"]


def test_apex_quality_filter_skips_a_zero_best_score():
    # A zero error has no meaningful ratio: every other expert would be excluded.
    sel = EnsembleSelector(apex_quality_ratio=2.0)
    pool = _experts([0.0, 10.0, 20.0])
    assert sel._filter_apex_pool(pool) == pool


@pytest.mark.parametrize("ratio", [0.5, float("nan")])
def test_apex_quality_ratio_below_one_is_rejected(ratio):
    with pytest.raises(ValueError):
        EnsembleSelector(apex_quality_ratio=ratio)
