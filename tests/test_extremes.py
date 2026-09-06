# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for the Generalized Pareto (peaks-over-threshold) tail model."""
import numpy as np
import pandas as pd
from scipy.stats import genpareto

from tam import StaticTAM
from tam.model.statistics.risk.extremes import GeneralizedParetoTail, fit_gpd_tail


def _heavy_tail_sample(rng, shape=0.3, scale=1.0):
    bulk = np.abs(rng.standard_normal(20000))
    tail = genpareto.rvs(shape, loc=0.0, scale=scale, size=4000, random_state=rng) + 2.0
    return np.concatenate([bulk, tail])


def test_gpd_recovers_shape_and_return_level_round_trips():
    rng = np.random.default_rng(0)
    fitted = GeneralizedParetoTail(threshold_quantile=0.95).fit(_heavy_tail_sample(rng, shape=0.3))
    assert abs(fitted.shape_ - 0.3) < 0.1
    level = fitted.return_level(0.001)
    assert abs(fitted.tail_probability([level])[0] - 0.001) < 1e-4


def test_tail_probability_and_score_are_monotone():
    rng = np.random.default_rng(1)
    fitted = GeneralizedParetoTail().fit(_heavy_tail_sample(rng))
    grid = fitted.threshold_ + np.array([0.1, 2.0, 6.0])
    assert np.all(np.diff(fitted.tail_probability(grid)) < 0)
    assert np.all(np.diff(fitted.anomaly_score(grid)) > 0)


def test_fit_gpd_tail_scores_extremes_far_above_bulk():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, 4000)
    frame = pd.DataFrame({"x": x, "y": np.exp(1.0 + 1.5 * x + (0.2 + 0.6 * x) * rng.standard_normal(4000))})
    model = StaticTAM(
        {"mu": "y ~ s(x)", "sigma": "y ~ s(x)"}, loss={"mu": "l2", "sigma": "gamma"}
    ).fit(frame)
    tail = fit_gpd_tail(model, frame, threshold_quantile=0.95)
    anomalies = frame.iloc[:20].copy()
    anomalies["y"] = anomalies["y"].to_numpy() * 30.0
    extreme_scores = tail.anomaly_score(np.abs(model.anomaly_score(anomalies)["z_score"].to_numpy()))
    bulk_scores = tail.anomaly_score(np.abs(model.anomaly_score(frame)["z_score"].to_numpy()))
    assert extreme_scores.mean() > 5.0 * max(bulk_scores.mean(), 1e-6)
