# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for the Gaussian copula over distributional StaticTAM margins."""
import numpy as np
import pandas as pd

from tam import StaticTAM, GaussianCopulaTAM

_LOSS = {"mu": "l2", "sigma": "gamma"}


def _margin(target):
    return StaticTAM({"mu": f"{target} ~ s(x)", "sigma": f"{target} ~ s(x)"}, loss=_LOSS)


def _correlated(rng, n=4000, rho=0.7):
    x = rng.uniform(0.0, 1.0, n)
    noise = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n) * 0.4
    return pd.DataFrame({
        "x": x,
        "y1": np.exp(1.0 + 1.2 * x + noise[:, 0]),
        "y2": np.exp(0.5 + 0.8 * x + noise[:, 1]),
    })


def _fit_copula(frame):
    return GaussianCopulaTAM({
        "y1": _margin("y1"),
        "y2": _margin("y2"),
    }).fit(frame)


def test_copula_recovers_dependence():
    frame = _correlated(np.random.default_rng(0), rho=0.7)
    copula = _fit_copula(frame)
    assert abs(copula.correlation_[0, 1] - 0.7) < 0.05


def test_joint_anomaly_flags_dependence_violation():
    rng = np.random.default_rng(1)
    frame = _correlated(rng, rho=0.7)
    copula = _fit_copula(frame)
    probe = frame.iloc[[0]].copy()
    probe["y1"] = probe["y1"].to_numpy() * np.exp(1.2)
    probe["y2"] = probe["y2"].to_numpy() * np.exp(-1.2)
    baseline = copula.joint_anomaly_score(frame)["joint_anomaly_score"].to_numpy()
    assert copula.joint_anomaly_score(probe)["joint_anomaly_score"].iloc[0] > np.quantile(baseline, 0.99)
