# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A prediction is a function of its own row: forecasting one row per group ("tomorrow, for every half-hour") gives the same value as
the same rows inside a long frame.

``s()`` took a frame with one row per group for the solver's one-row memory probe and rebuilt its knots from that single point
instead of using the trained ones (up to 16,000 MW off on the national load).
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta

FORMULAS = [
    "y ~ s(x, k=6)",
    "y ~ s(x, k=6) + c(k)",
    "y ~ f(x)",
    "y ~ w(x)",
    "y ~ rbf(x, n_centers=8)",
    "y ~ t(x)",
    "y ~ l(x) + c(k)",
]
N_DAYS = 40


def _data(seed=0, groups=(0, 1, 2)):
    rng = np.random.default_rng(seed)
    frames = []
    for j, g in enumerate(groups):
        x = rng.uniform(-2, 2, N_DAYS)
        frames.append(pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=N_DAYS, freq="D"), "g": g, "x": x,
            "k": rng.integers(0, 3, N_DAYS), "y": np.sin(2 * x) + j + 0.1 * rng.standard_normal(N_DAYS)}))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("formula", FORMULAS)
def test_one_row_per_group_equals_the_same_rows_in_a_long_frame(formula):
    df = _data()
    model = ta.StaticTAM(formula=formula, group_col="g", date_col="date").fit(df)
    test = _data(seed=1)
    test["x"] = test["x"] * 1.1                       # slightly beyond the training range on some rows
    long = model.predict(test)
    day = test.groupby("g").cumcount() == 17
    one = model.predict(test[day])
    np.testing.assert_allclose(one["Estimatedy"].to_numpy(), long.loc[day, "Estimatedy"].to_numpy(), rtol=1e-9, atol=1e-9)


def test_one_row_without_groups_equals_the_same_row_in_a_long_frame():
    df = _data(groups=(0,))
    model = ta.StaticTAM(formula="y ~ s(x, k=6)", date_col="date").fit(df)
    test = _data(seed=1, groups=(0,))
    long = model.predict(test)
    one = model.predict(test.iloc[[17]])
    np.testing.assert_allclose(one["Estimatedy"].to_numpy(), long.loc[[17], "Estimatedy"].to_numpy(), rtol=1e-9, atol=1e-9)


def test_decomposition_of_one_row_per_group_equals_the_long_frame():
    df = _data()
    model = ta.StaticTAM(formula="y ~ s(x, k=6) + c(k)", group_col="g", date_col="date").fit(df)
    test = _data(seed=2)
    long = model.decompose_prediction(test)
    day = test.groupby("g").cumcount() == 5
    one = model.decompose_prediction(test[day])
    effects = [c for c in long.columns if c.startswith("effect_")]
    np.testing.assert_allclose(one[effects].to_numpy(), long.loc[day, effects].to_numpy(), rtol=1e-9, atol=1e-9)


def test_a_single_out_of_range_value_extrapolates_like_several():
    """With the default linear extrapolation, the slope of the out-of-range values is computed on a tensor holding only those values:
    exactly one of them has one element, which looked like the memory probe (knots rebuilt from that single point, up to millions of MW off).
    Tested on the effect, on normalised features: one value beyond [-1, 1] gives the same basis as the same value next to a second one."""
    import torch
    from tam.model.spectrum._spline import SplineEffect

    df = _data()
    model = ta.StaticTAM(formula="y ~ s(x, k=6)", group_col="g", date_col="date").fit(df)
    effect = next(e for e in model.effects_list_ if isinstance(e, SplineEffect))
    assert effect.extrapolate == "linear"
    x = torch.linspace(-0.9, 0.9, 20, dtype=torch.get_default_dtype()).repeat(2, 1)        # (groups, rows), all inside
    one = x.clone()
    one[0, 5] = 1.05                                                                       # exactly one value beyond the boundary
    two = one.clone()
    two[1, 3] = 1.2                                                                        # and a second one elsewhere
    a, b = effect.transform(one)[0, 5], effect.transform(two)[0, 5]
    assert torch.isfinite(a).all() and a.abs().max() < 1e3
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
