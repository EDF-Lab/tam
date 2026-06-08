r"""
Unit tests for ``tam.model.safety.SafetyTAM`` (Conformal Prediction).

Covers calibration, the finite-sample quantile, both static (split conformal)
and adaptive (ACI) interval modes, and the error guards.
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
        model.predict_intervals(np.array([1.0, 2.0]), method="static")


def test_static_intervals_have_constant_width(calibrated_model):
    y_pred = np.full(10, 50.0)
    df = calibrated_model.predict_intervals(y_pred, method="static")

    assert list(df.columns[:5]) == ["Predicted", "Lower", "Upper", "Alpha_t", "Width"]
    # Split conformal uses a single fixed width and the target alpha throughout.
    assert np.allclose(df["Width"], df["Width"].iloc[0])
    assert np.allclose(df["Alpha_t"], 0.1)
    assert (df["Upper"] >= df["Lower"]).all()


def test_aci_requires_online_truth(calibrated_model):
    with pytest.raises(ValueError, match="y_true_online"):
        calibrated_model.predict_intervals(np.full(5, 50.0), method="aci")


def test_aci_adapts_alpha_and_reports_coverage(calibrated_model):
    rng = np.random.default_rng(1)
    y_pred = rng.normal(50, 5, 60)
    y_true_online = y_pred + rng.normal(0, 2, 60)

    df = calibrated_model.predict_intervals(
        y_pred, y_true_online=y_true_online, method="aci", gamma=0.05
    )
    assert "Covered" in df.columns
    assert "Actual" in df.columns
    # The adaptive risk level stays within the documented safety bounds.
    assert (df["Alpha_t"] >= 0.001).all() and (df["Alpha_t"] <= 0.999).all()
    # Empirical coverage should be near the 90% target on well-behaved data.
    assert df["Covered"].mean() > 0.7


def test_invalid_method_raises(calibrated_model):
    with pytest.raises(ValueError, match="'static' or 'aci'"):
        calibrated_model.predict_intervals(np.full(5, 50.0), method="bayesian")
