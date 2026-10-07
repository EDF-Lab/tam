# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
``rbf()`` centres come from a local generator seeded by the formula (``seed``, default 42).

The fit is reproducible whatever other code consumed random numbers before, and fitting leaves the global torch random state
untouched.
"""

import numpy as np
import pandas as pd
import torch

import tam as ta


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x": x,
                         "y": np.sin(x) + rng.normal(0, 0.1, n)})


def _fit(formula, df):
    model = ta.StaticTAM(formula=formula, date_col="date").fit(df)
    rbf = [e for e in model.effects_list_ if e.effect_type.startswith("rbf")][0]
    return model, rbf


def test_two_fits_are_identical_even_after_other_random_draws():
    df = _data()
    torch.manual_seed(0)
    m1, r1 = _fit("y ~ rbf(x, n_centers=20)", df)
    torch.rand(1000)  # other code consuming the global generator
    m2, r2 = _fit("y ~ rbf(x, n_centers=20)", df)
    assert torch.equal(r1.centers, r2.centers)
    assert torch.equal(m1.coefficients_, m2.coefficients_)


def test_fitting_leaves_the_global_random_state_unchanged():
    df = _data()
    state = torch.get_rng_state()
    _fit("y ~ rbf(x, n_centers=20)", df)
    assert torch.equal(torch.get_rng_state(), state)


def test_the_default_seed_is_42_and_other_seeds_give_other_centres():
    df = _data()
    _, default = _fit("y ~ rbf(x, n_centers=20)", df)
    _, seed42 = _fit("y ~ rbf(x, n_centers=20, seed=42)", df)
    _, seed7 = _fit("y ~ rbf(x, n_centers=20, seed=7)", df)
    assert default.seed == 42
    assert torch.equal(default.centers, seed42.centers)
    assert not torch.equal(default.centers, seed7.centers)
