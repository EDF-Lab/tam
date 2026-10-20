# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``n(..., orthogonalize=True)`` and ``t(..., orthogonalize=True)``: the flexible block (neural features, trees) is projected on the
orthogonal complement of the rest of the design (intercept, splines, linear terms...), so that it can only explain what the interpretable
terms cannot. The projection is estimated once on the training rows and kept for the forecast.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.common.utils import _balance_groups, _ensure_dummies
from tam.model.spectrum import NeuralEffect, TreeEffect, build_phi_from_effects


def _data(n=1500, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    y = np.sin(2.5 * x1) + x2 ** 2 + 1.5 * x1 * x2 + noise * rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _design(model, frame):
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    x, _, _ = model._prepare_data(balanced, target_col="y")
    return model._build_design_matrix(x)[0]


def _split_blocks(model):
    sizes = [e.get_n_coeffs() for e in model.effects_list_]
    flagged = [getattr(e, "orthogonalize", False) for e in model.effects_list_]
    return sizes, flagged


@pytest.mark.parametrize("flexible", ["n(x1, others='x2', n_neurons=60, ap=-6, orthogonalize=True)",
                                      "t(x1, others='x2', n_trees=20, max_depth=3, ap=-6, orthogonalize=True)"])
def test_the_flexible_block_is_orthogonal_to_the_rest_of_the_design(flexible):
    frame = _data()
    model = ta.StaticTAM(f"y ~ s(x1, k=8) + s(x2, k=8) + {flexible}").fit(frame)
    phi = _design(model, frame).double()
    sizes, flagged = _split_blocks(model)
    edges = np.cumsum([0] + sizes)
    deep = torch.cat([phi[:, edges[i]:edges[i + 1]] for i, f in enumerate(flagged) if f], dim=1)
    base = torch.cat([phi[:, edges[i]:edges[i + 1]] for i, f in enumerate(flagged) if not f], dim=1)
    cross = base.T @ deep
    assert float(cross.abs().max()) < 1e-8 * float(base.abs().max() * deep.abs().max()) * len(frame)


def test_without_the_flag_nothing_changes():
    frame = _data()
    a = ta.StaticTAM("y ~ s(x1, k=8) + n(x1, others='x2', n_neurons=40, ap=-6)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    b = ta.StaticTAM("y ~ s(x1, k=8) + n(x1, others='x2', n_neurons=40, ap=-6, orthogonalize=False)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    assert np.array_equal(a, b)


def test_the_splines_keep_the_main_effects_when_the_flexible_block_is_orthogonal():
    frame, test = _data(seed=1), _data(seed=2, noise=0.0)
    errors = {}
    for flag in (False, True):
        model = ta.StaticTAM(f"y ~ s(x1, k=10, ap=-6) + s(x2, k=10, ap=-6) + n(x1, others='x2', n_neurons=150, ap=-8, orthogonalize={flag})").fit(frame)
        parts = model.decompose_prediction(test)
        column = "effect_s_x1"
        main = np.sin(2.5 * test["x1"].to_numpy())
        got = parts[column].to_numpy()
        errors[flag] = float(np.sqrt(np.mean(((got - got.mean()) - (main - main.mean())) ** 2)))
    assert errors[True] < 0.5 * errors[False]


def test_the_total_fit_is_not_worse_when_the_block_is_orthogonalised():
    frame, test = _data(seed=3), _data(seed=4, noise=0.0)
    truth = test["y"].to_numpy()
    rmse = {}
    for flag in (False, True):
        model = ta.StaticTAM(f"y ~ s(x1, k=10) + s(x2, k=10) + n(x1, others='x2', n_neurons=100, ap=-6, orthogonalize={flag})").fit(frame)
        rmse[flag] = float(np.sqrt(np.mean((model.predict(test)["Estimatedy"].to_numpy() - truth) ** 2)))
    assert rmse[True] < 1.15 * rmse[False] + 1e-3


def test_the_projection_is_the_same_in_a_batch_and_row_by_row():
    frame, test = _data(seed=5), _data(n=40, seed=6)
    model = ta.StaticTAM("y ~ s(x1, k=8) + n(x1, others='x2', n_neurons=40, ap=-6, orthogonalize=True)").fit(frame)
    batch = model.predict(test)["Estimatedy"].to_numpy()
    single = np.array([model.predict(test.iloc[[i]])["Estimatedy"].iloc[0] for i in range(len(test))])
    assert np.allclose(batch, single, atol=1e-9)


def test_the_flag_is_reported_by_the_effect():
    model = ta.StaticTAM("y ~ s(x1, k=8) + n(x1, others='x2', n_neurons=20, orthogonalize=True)").fit(_data(300))
    assert [type(e) for e in model.effects_list_ if getattr(e, "orthogonalize", False)] == [NeuralEffect]
