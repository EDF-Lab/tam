# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._pid.PIDEffect``."""

import pytest
import torch

import tam
from tam.model.spectrum import PIDEffect


def test_pid_contract(normalized, penalty_shape):
    effect = PIDEffect("x", window=3, lambda_p=1.0, d_penalty_multiplier=10.0, extrapolate="continue")
    assert effect.get_n_coeffs() == 3  # proportional, integral, derivative

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 3)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (3, 3)


def test_pid_penalty_boosts_derivative_term():
    effect = PIDEffect("x", window=3, lambda_p=1.0, d_penalty_multiplier=10.0, extrapolate="continue")
    P = effect.build_penalty_matrix()
    diag = torch.diagonal(P)
    # Derivative (index 2) is penalized harder than P and I.
    assert diag[2] == 10.0 * diag[0]


def test_pid_zero_window_uses_full_cumulative_integral(normalized):
    """window=0 collapses the rolling-mean integral to the raw cumulative sum."""
    effect = PIDEffect("x", window=0, lambda_p=1.0, d_penalty_multiplier=1.0, extrapolate="continue")
    phi = effect.build_feature_map(normalized(2, 8))
    assert phi.shape == (2, 8, 3)
    assert torch.isfinite(phi).all()


def test_pid_negative_window_raises():
    with pytest.raises(ValueError, match="non-negative"):
        PIDEffect("x", window=-1, lambda_p=1.0, d_penalty_multiplier=1.0, extrapolate="continue")


def _derivative_signal(n=1500, n_train=1000):
    """A stationary AR(1) ``x`` whose derivative carries the information: ``y = x + 5 dx + noise``."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    y = x + 5 * np.diff(x, prepend=x[0]) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=n, freq="h"), "grp": "G", "x": x, "y": y})
    return df, n_train


def _test_rmse(formula):
    import numpy as np

    df, n_train = _derivative_signal()
    model = tam.StaticTAM(formula=formula, group_col="grp", date_col="timestamp")
    model.fit(df.iloc[:n_train])
    pred = model.predict(df)["Estimatedy"].to_numpy()[n_train:]
    return float(np.sqrt(np.mean((pred - df["y"].to_numpy()[n_train:]) ** 2)))


def test_pid_beats_a_linear_term_when_the_derivative_carries_the_signal():
    assert _test_rmse("y ~ pid(x, w=6, d_pen=0)") < 0.2 * _test_rmse("y ~ l(x)")


def test_d_pen_is_hidden_by_the_default_penalty():
    """With the default (negligible) penalty, d_pen multiplies a ridge that is already ~0: the forecasts agree."""
    low, high = _test_rmse("y ~ pid(x, w=6, d_pen=0)"), _test_rmse("y ~ pid(x, w=6, d_pen=100)")
    assert abs(low - high) < 0.01 * low


def test_d_pen_acts_when_the_penalty_is_explicit():
    """With an explicit penalty (ap=-2) a large d_pen penalises the derivative out and the forecast gets worse."""
    low, high = _test_rmse("y ~ pid(x, w=6, d_pen=0, ap=-2)"), _test_rmse("y ~ pid(x, w=6, d_pen=100, ap=-2)")
    assert high > 1.1 * low
