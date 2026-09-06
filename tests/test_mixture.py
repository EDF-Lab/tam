# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for the mixture mode of :class:`StaticTAM` (finite Gaussian mixture fitted by EM)."""
import numpy as np
import pandas as pd

from tam import StaticTAM


def _bimodal(rng, n=3000, separation=2.0):
    x = rng.uniform(0.0, 1.0, n)
    regime = rng.integers(0, 2, n)
    log_target = (1.0 + 1.5 * x) + regime * separation + 0.15 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": np.exp(log_target)}), regime


def test_mixture_recovers_two_regimes():
    rng = np.random.default_rng(0)
    frame, regime = _bimodal(rng, separation=2.0)
    model = StaticTAM("y ~ s(x)", mixture_components=2, mixture_kwargs={"seed": 0}).fit(frame)
    component_means = np.log(model.component_means(frame))
    assert abs(np.median(np.abs(component_means[:, 0] - component_means[:, 1])) - 2.0) < 0.4
    predicted_regime = np.argmax(model.responsibilities(frame), axis=1)
    accuracy = max(np.mean(predicted_regime == regime), np.mean(predicted_regime == (1 - regime)))
    assert accuracy > 0.9


def test_mixture_beats_single_component_likelihood():
    rng = np.random.default_rng(1)
    frame, _ = _bimodal(rng)
    two = StaticTAM("y ~ s(x)", mixture_components=2, mixture_kwargs={"seed": 0}).fit(frame)
    one = StaticTAM("y ~ s(x)", mixture_components=1, mixture_kwargs={"seed": 0}).fit(frame)
    assert two.log_likelihood_ > one.log_likelihood_
    assert abs(two.mixing_weights_.sum() - 1.0) < 1e-9


def test_mixture_anomaly_score_flags_off_manifold_points():
    rng = np.random.default_rng(2)
    frame, _ = _bimodal(rng)
    model = StaticTAM("y ~ s(x)", mixture_components=2, mixture_kwargs={"seed": 0}).fit(frame)
    between = frame.iloc[:50].copy()
    between["y"] = np.exp(np.log(between["y"].to_numpy()) + 1.0)
    assert model.anomaly_score(between).mean() > np.quantile(model.anomaly_score(frame), 0.9)
