# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Tests for ``tam.model.statistics.risk.aci``, the Adaptive Conformal Inference control loop, extracted from
the static ``SafetyTAM`` engine. ACI drives *any* calibrated radius provider (here, a ``SafetyTAM``'s
finite-sample ``conformal_quantile``) and schedules the risk level online.
"""
import numpy as np
import pytest

from tam.model.safety import SafetyTAM
from tam.model.statistics.risk.aci import (
    adaptive_conformal_intervals,
    adaptive_conformal_scores,
    effective_alpha,
    update_risk_level,
)


def _calibrated(seed, n=200):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(50, 5, n)
    y_pred = y_true + rng.normal(0, 2, n)
    return SafetyTAM(alpha=0.1).calibrate(y_true, y_pred)


# --- the single-step SGD update ------------------------------------------- #
def test_update_risk_level_direction_and_unbounded_integrator():
    # covered (error=0) raises alpha (tighter next interval); miscoverage (error=1) lowers it (wider)
    assert update_risk_level(0.10, 0.0, 0.10, 0.05) > 0.10
    assert update_risk_level(0.10, 1.0, 0.10, 0.05) < 0.10
    # the raw integrator is intentionally unbounded (anti-windup under sustained drift)...
    assert update_risk_level(0.02, 1.0, 0.10, 0.5) < 0.0
    # ...and is clamped only where it is used, by effective_alpha.
    assert effective_alpha(-0.5) == pytest.approx(1e-3)
    assert effective_alpha(5.0) == pytest.approx(0.999)
    assert effective_alpha(0.1) == pytest.approx(0.1)


# --- the score-stream loop ------------------------------------------------ #
def test_adaptive_conformal_scores_adapts_radius_and_holds_coverage():
    rng = np.random.default_rng(2)
    model = SafetyTAM(alpha=0.1).calibrate_scores(np.abs(rng.standard_normal(5000)))
    drifting = np.abs(rng.standard_normal(4000)) * np.linspace(1.0, 1.8, 4000)
    radii, alphas = adaptive_conformal_scores(model.conformal_quantile, drifting, model.alpha_target, gamma=0.03)
    assert radii[-200:].mean() > radii[:200].mean()          # radius widens as the stream drifts up
    assert abs(np.mean(drifting <= radii) - 0.9) < 0.05       # long-run coverage tracks the 90% target
    assert (alphas >= 1e-3).all() and (alphas <= 0.999).all()


# --- the streaming interval loop ------------------------------------------ #
def test_adaptive_conformal_intervals_requires_online_truth():
    model = _calibrated(0)
    with pytest.raises(ValueError, match="y_true_online"):
        adaptive_conformal_intervals(model.conformal_quantile, np.full(5, 50.0), None, model.alpha_target)


def test_adaptive_conformal_intervals_adapts_alpha_and_reports_coverage():
    model = _calibrated(0)
    rng = np.random.default_rng(1)
    y_pred = rng.normal(50, 5, 60)
    y_true_online = y_pred + rng.normal(0, 2, 60)
    df = adaptive_conformal_intervals(
        model.conformal_quantile, y_pred, y_true_online, model.alpha_target, gamma=0.05
    )
    assert "Covered" in df.columns and "Actual" in df.columns
    assert (df["Alpha_t"] >= 1e-3).all() and (df["Alpha_t"] <= 0.999).all()
    assert df["Covered"].mean() > 0.7


def test_adaptive_conformal_intervals_normalized_by_scale():
    model = _calibrated(0)
    y_pred = np.zeros(4)
    y_true = np.zeros(4)
    scale = np.array([1.0, 2.0, 4.0, 8.0])
    df = adaptive_conformal_intervals(model.conformal_quantile, y_pred, y_true, model.alpha_target, scale=scale)
    # with a covered stream the level (and hence the base radius) is monotone, so width scales with sigma
    assert np.all(np.diff(df["Width"].to_numpy()) > 0)
