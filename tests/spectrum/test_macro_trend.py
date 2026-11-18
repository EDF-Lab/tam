# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the Hodrick-Prescott macro-trend effect (hp_trend)."""

import numpy as np
import pytest
import pandas as pd
import torch

from tam import StaticTAM
from tam.model.spectrum._inertia import MacroTrendEffect


def test_macro_trend_natural_linear_extrapolation():
    """MacroTrendEffect extrapolates strictly linearly outside the training bounds."""
    hp = MacroTrendEffect("time", k=10)
    assert hp.get_n_coeffs() == 13

    t_train = torch.linspace(0.0, 100.0, 101, dtype=torch.float64)
    phi_train = hp.build_feature_map(t_train)
    assert phi_train.shape == (101, 13)

    t_beyond = torch.linspace(101.0, 150.0, 50, dtype=torch.float64)
    phi_beyond = hp.build_feature_map(t_beyond)

    torch.manual_seed(42)
    theta = torch.randn(13, dtype=torch.float64)
    f_beyond = phi_beyond @ theta
    d2 = torch.diff(f_beyond, n=2)
    np.testing.assert_allclose(d2.numpy(), 0.0, atol=1e-8)

    df_train = pd.DataFrame({"time": np.arange(100, dtype=float), "y": np.linspace(50.0, 150.0, 100) + np.random.randn(100)})
    df_test = pd.DataFrame({"time": np.arange(100, 150, dtype=float)})
    m = StaticTAM("y ~ hp_trend(time, k=8, ap=-2)")
    m.fit(df_train)
    p = m.predict(df_test)
    assert not p["Estimatedy"].isna().any()


def test_macro_trend_bounds_are_those_of_the_last_initialize():
    """A second initialize replaces the range of the trend axis."""
    hp = MacroTrendEffect("time", k=8)
    hp.initialize(torch.linspace(0.0, 10.0, 50, dtype=torch.float64))
    first = hp._x_bounds
    hp.initialize(torch.linspace(100.0, 200.0, 50, dtype=torch.float64))
    assert first[1] < 20 and hp._x_bounds[0] == pytest.approx(100.0) and hp._x_bounds[1] == pytest.approx(200.0, abs=1e-3)
    phi = hp.build_feature_map(torch.tensor([[100.0, 150.0, 200.0]], dtype=torch.float64))
    assert phi.shape == (1, 3, hp.get_n_coeffs())
    assert hp._x_bounds[0] == pytest.approx(100.0)          # building does not move the frozen range


def test_macro_trend_is_pointwise_across_groups():
    """The trend reads one value at a time: a group's columns do not depend on the other groups."""
    hp = MacroTrendEffect("time", k=8)
    x = torch.rand(3, 40, dtype=torch.float64)
    hp.initialize(x)
    whole = hp.build_feature_map(x)
    assert torch.allclose(hp.build_feature_map(x[1:2])[0], whole[1], atol=1e-12)


def test_macro_trend_is_in_the_load_forecasting_shim():
    from tam.model.spectrum import _load_forecasting as lf

    assert lf.MacroTrendEffect is MacroTrendEffect and lf.SmoothHingeEffect.__name__ == "DegreeDayEffect"
    from tam.model.spectrum._factory import extrapolating_features

    assert extrapolating_features([MacroTrendEffect("time")]) == {"time"}
