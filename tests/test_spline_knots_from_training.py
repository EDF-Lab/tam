# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
The knots of a spline come from the training tensor, never from the solver's memory probe (the first row of every group).

When the first row of every group sat on the upper edge of its range, the probe reached the extrapolation branch of ``s()``, whose
step back from the boundary is a call on a few values that was taken for real data: the knots were cached from it, all equal to 1,
and every later forecast outside the trained range was off by orders of magnitude.
"""

import numpy as np
import pandas as pd
import torch

import tam as ta
from tam.model.spectrum import SplineEffect, initialize_effects


def _groups(n_groups=3, n=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        z = rng.uniform(-1, 1, n)
        frames.append(pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="D"), "g": g, "z": z,
                                    "y": np.sin(2 * z) + 0.05 * rng.standard_normal(n)}))
    return pd.concat(frames, ignore_index=True)


def _spline(model):
    return next(e for e in model.effects_list_ if isinstance(e, SplineEffect))


def test_knots_are_set_from_the_training_tensor_before_any_design_matrix_is_built():
    df = _groups()
    model = ta.StaticTAM(formula="y ~ s(z, k=6)", group_col="g", date_col="date")
    model._prepare_data(df, target_col="y")
    knots = _spline(model)._cached_knots
    assert knots is not None
    assert float(knots.min()) == -1.0 and float(knots.max()) == 1.0


def test_a_probe_on_the_boundary_does_not_decide_the_knots():
    """One value per group sitting on the edge (and a hair beyond it) goes through the extrapolation step back."""
    effect = SplineEffect("z", n_knots=6, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="linear")
    training = torch.linspace(-1, 1, 30, dtype=torch.get_default_dtype()).repeat(3, 1).unsqueeze(-1)
    initialize_effects(training, [effect], feature_columns=["z"])
    before = effect._cached_knots.clone()
    probe = torch.full((3, 1), 1.0 + 1e-12, dtype=torch.get_default_dtype())
    effect.transform(probe)
    torch.testing.assert_close(effect._cached_knots, before)
    assert float(effect._cached_knots.min()) == -1.0 and float(effect._cached_knots.max()) == 1.0

