# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for conformal calibration of a distributional StaticTAM (CQR, conformal p-values, ACI)."""
import numpy as np
import pandas as pd

from tam import StaticTAM, ConformalDistributionalTAM

_LOSS = {"mu": "l2", "sigma": "gamma"}


def _lognormal(rng, n, heavy=True):
    x = rng.uniform(0.0, 1.0, n)
    mu, sigma = 1.0 + 1.5 * x, 0.2 + 0.6 * x
    noise = rng.standard_t(2.5, n) / np.sqrt(2.5 / 0.5) if heavy else rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": np.exp(mu + sigma * noise)})


def _fitted_conformal(rng, alpha=0.1):
    model = StaticTAM(
        {"mu": "y ~ s(x)", "sigma": "y ~ s(x)"}, loss=_LOSS, dist_kwargs={"tail_family": "normal"}
    ).fit(_lognormal(rng, 4000))
    return ConformalDistributionalTAM(model, alpha=alpha).calibrate(_lognormal(rng, 3000))


def test_cqr_restores_coverage_under_wrong_tail():
    rng = np.random.default_rng(0)
    conformal = _fitted_conformal(rng)
    test = _lognormal(rng, 4000)
    interval = conformal.predict_interval(test)
    covered = np.mean((test["y"].to_numpy() >= interval["lower"].to_numpy())
                      & (test["y"].to_numpy() <= interval["upper"].to_numpy()))
    assert abs(covered - 0.9) < 0.025


def test_conformal_pvalue_false_positive_rate_matches_alpha():
    rng = np.random.default_rng(1)
    conformal = _fitted_conformal(rng)
    clean = _lognormal(rng, 5000)
    assert abs(np.mean(conformal.conformal_pvalue(clean) < 0.1) - 0.1) < 0.02


def test_conformal_flags_injected_anomalies():
    rng = np.random.default_rng(2)
    conformal = _fitted_conformal(rng)
    anomalies = _lognormal(rng, 100)
    anomalies["y"] = anomalies["y"].to_numpy() * 15.0
    assert np.mean(conformal.conformal_pvalue(anomalies) < 0.1) > 0.9


def test_aci_maintains_coverage_under_drift():
    rng = np.random.default_rng(3)
    conformal = _fitted_conformal(rng)
    drift = _lognormal(rng, 4000).reset_index(drop=True)
    drift["y"] = drift["y"].to_numpy() * np.exp(np.linspace(0.0, 1.5, len(drift)))
    intervals = conformal.aci_intervals(drift, gamma=0.03)
    covered = np.mean((drift["y"].to_numpy() >= intervals["lower"].to_numpy())
                      & (drift["y"].to_numpy() <= intervals["upper"].to_numpy()))
    assert abs(covered - 0.9) < 0.03
