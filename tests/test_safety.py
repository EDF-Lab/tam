# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

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



# --- verbose and multi-level quantiles ------------------------------------------------------------------------------
_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_calibrate_prints_by_default_and_verbose_false_is_silent(capsys):
    y = np.random.default_rng(0).standard_normal(100)
    SafetyTAM(alpha=0.1).calibrate(y, np.zeros(100))
    assert "calibrated on 100 samples" in capsys.readouterr().out
    SafetyTAM(alpha=0.1).calibrate(y, np.zeros(100), verbose=False)
    assert capsys.readouterr().out == ""


def test_symmetric_quantiles_match_the_intervals():
    rng = np.random.default_rng(1)
    y_pred, scale = rng.normal(size=50), rng.uniform(0.5, 2.0, 50)
    engine = SafetyTAM(alpha=0.2).calibrate(rng.standard_normal(500), np.zeros(500), verbose=False)
    table = engine.predict_quantiles(y_pred, _LEVELS, scale=scale)
    assert list(table.columns) == [f"q{t}" for t in _LEVELS]
    band = engine.predict_intervals(y_pred, scale=scale)  # alpha 0.2 -> levels 0.1 and 0.9
    np.testing.assert_allclose(table["q0.1"], band["Lower"])
    np.testing.assert_allclose(table["q0.9"], band["Upper"])
    np.testing.assert_allclose(table["q0.5"], y_pred)


def test_quantiles_are_non_decreasing_across_levels():
    rng = np.random.default_rng(2)
    engine = SafetyTAM().calibrate(rng.lognormal(size=300), np.ones(300), verbose=False)
    for signed in (False, True):
        table = engine.predict_quantiles(rng.normal(size=40), _LEVELS, signed=signed).to_numpy()
        assert (np.diff(table, axis=1) >= 0).all()


def test_signed_quantiles_cover_each_tail_of_skewed_errors():
    # Lognormal errors are skewed: the symmetric (absolute-score) band is too wide on the short side and too narrow on
    # the long one; signed scores calibrate each tail on its own.
    rng = np.random.default_rng(3)
    errors = lambda n: rng.lognormal(0.0, 0.8, n) - np.exp(0.32)  # noqa: E731 - centred at the mean
    engine = SafetyTAM().calibrate(errors(4000), np.zeros(4000), verbose=False)
    future = errors(20000)
    for signed, tolerance in ((True, 0.02), (False, None)):
        table = engine.predict_quantiles(np.zeros_like(future), [0.1, 0.9], signed=signed)
        below_lower = float(np.mean(future < table["q0.1"]))
        above_upper = float(np.mean(future > table["q0.9"]))
        if signed:
            assert abs(below_lower - 0.1) < tolerance and abs(above_upper - 0.1) < tolerance
        else:
            assert below_lower < 0.08 and above_upper > 0.12  # the symmetric band misplaces both tails (about 6.5 % and 13 %)


def test_signed_quantiles_need_signed_scores():
    engine = SafetyTAM().calibrate_scores(np.abs(np.random.default_rng(4).standard_normal(100)))
    with pytest.raises(ValueError, match="signed"):
        engine.predict_quantiles(np.zeros(3), _LEVELS, signed=True)
