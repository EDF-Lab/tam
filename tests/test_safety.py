# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.safety.SafetyTAM``, the **static** split-conformal engine.

Covers calibration, the finite-sample quantile, p-values, and static (constant-width / normalized) interval
prediction. The streaming Adaptive Conformal Inference loop now lives in ``tam.model.statistics.risk.aci`` and
is tested in ``test_aci.py``.
"""

import numpy as np
import pytest

from tam.model.safety import SafetyTAM


@pytest.fixture
def calibrated_model():
    rng = np.random.default_rng(0)
    model = SafetyTAM(alpha=0.1)
    y_true = rng.normal(50, 5, 200)
    y_pred = y_true + rng.normal(0, 2, 200)
    model.calibrate(y_true, y_pred)
    return model


def test_calibrate_stores_absolute_residuals():
    model = SafetyTAM(alpha=0.1)
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    model.calibrate(y_true, y_pred)
    np.testing.assert_allclose(model.residuals_calib_, [2.0, 2.0, 3.0])


def test_predict_intervals_requires_calibration():
    model = SafetyTAM(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict_intervals(np.array([1.0, 2.0]))


def test_static_intervals_have_constant_width(calibrated_model):
    y_pred = np.full(10, 50.0)
    df = calibrated_model.predict_intervals(y_pred)

    assert list(df.columns[:5]) == ["Predicted", "Lower", "Upper", "Alpha_t", "Width"]
    # Split conformal uses a single fixed width and the target alpha throughout.
    assert np.allclose(df["Width"], df["Width"].iloc[0])
    assert np.allclose(df["Alpha_t"], 0.1)
    assert (df["Upper"] >= df["Lower"]).all()


def test_static_intervals_report_coverage_when_truth_supplied(calibrated_model):
    rng = np.random.default_rng(5)
    y_pred = rng.normal(50, 5, 400)
    y_true_online = y_pred + rng.normal(0, 2, 400)
    df = calibrated_model.predict_intervals(y_pred, y_true_online=y_true_online)
    assert "Covered" in df.columns and "Actual" in df.columns
    assert df["Covered"].mean() > 0.85  # ~90% target on well-behaved i.i.d. data


def test_aci_logic_is_no_longer_on_the_static_engine(calibrated_model):
    # ACI was moved out of SafetyTAM into tam.model.statistics.risk.aci.
    assert not hasattr(calibrated_model, "aci_scores")
    import inspect
    assert "method" not in inspect.signature(calibrated_model.predict_intervals).parameters


def test_calibrate_scores_and_conformal_quantile():
    rng = np.random.default_rng(0)
    model = SafetyTAM(alpha=0.1).calibrate_scores(np.abs(rng.standard_normal(20000)))
    assert abs(model.conformal_quantile(0.1) - 1.645) < 0.05


def test_pvalue_false_positive_rate_matches_alpha():
    rng = np.random.default_rng(1)
    model = SafetyTAM(alpha=0.1).calibrate_scores(np.abs(rng.standard_normal(5000)))
    pvalues = model.pvalue(np.abs(rng.standard_normal(5000)))
    assert abs(np.mean(pvalues < 0.1) - 0.1) < 0.02


def test_normalized_intervals_scale_with_sigma():
    rng = np.random.default_rng(3)
    scale = rng.uniform(1.0, 4.0, 2000)
    model = SafetyTAM(alpha=0.1).calibrate(np.abs(rng.standard_normal(2000)), np.zeros(2000), scale=scale)
    test_scale = np.array([1.0, 2.0, 4.0, 1.0, 2.0])
    intervals = model.predict_intervals(np.zeros(5), scale=test_scale)
    ratio = intervals["Width"].to_numpy() / test_scale
    assert np.allclose(ratio, ratio[0])
