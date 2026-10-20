# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``c(region, topo='graph', coords='x:y', k_neighbors=6)``: a Markov random field over the levels. The neighbours are the k nearest
centroids (mean of the coordinates of the rows of each level) and the penalty is the graph Laplacian, so a region with few observations
borrows strength from the regions around it. A region with no training row has no centroid and falls back to the zero effect.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import CategoricalEffect

SIDE = 6


def _regions(rows_per_region=3, seed=0, noise=0.6):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
    cx, cy = gx.ravel().astype(float), gy.ravel().astype(float)
    code = np.repeat(np.arange(SIDE * SIDE), rows_per_region)
    truth = np.sin(cx / 2.0) + np.cos(cy / 2.5)
    x = cx[code] + 0.05 * rng.normal(size=code.size)
    y = cy[code] + 0.05 * rng.normal(size=code.size)
    return pd.DataFrame({"region": code, "x": x, "y": y, "target": truth[code] + noise * rng.normal(size=code.size)}), truth


def _effect(n_cat=SIDE * SIDE, **kwargs):
    return CategoricalEffect("region", n_cat, "graph", 1e-2, 1, "continue", coords=["x", "y"], **kwargs)


def _grid_tensor(rows_per_region=3):
    frame, _ = _regions(rows_per_region)
    n_cat = SIDE * SIDE
    level = frame["region"].to_numpy() / (n_cat - 1) * 2 - 1
    x = (frame["x"].to_numpy() - (SIDE - 1) / 2) / ((SIDE - 1) / 2)
    y = (frame["y"].to_numpy() - (SIDE - 1) / 2) / ((SIDE - 1) / 2)
    return torch.tensor(np.stack([level, x, y], axis=-1)).unsqueeze(0)


def test_graph_needs_coordinates():
    with pytest.raises(ValueError, match="coords"):
        CategoricalEffect("region", 9, "graph", 1e-2, 1, "continue")


def test_the_formula_reads_the_coordinate_columns():
    model = ta.StaticTAM("target ~ c(region, topo='graph', coords='x:y', k_neighbors=4)")
    assert model.features_config_["features"] == ["region", "x", "y"]


def test_the_penalty_is_a_graph_laplacian():
    effect = _effect(k_neighbors=4)
    effect.initialize(_grid_tensor())
    penalty = effect.build_penalty_matrix() / effect.lambda_p
    n = penalty.shape[0]
    assert torch.allclose(penalty, penalty.T, atol=1e-12)
    assert float(torch.linalg.eigvalsh(penalty).min()) > 0.0  # the small ridge makes it positive definite
    laplacian = penalty - 1e-4 * torch.eye(n, dtype=penalty.dtype)
    assert torch.allclose(laplacian @ torch.ones(n, dtype=penalty.dtype), torch.zeros(n, dtype=penalty.dtype), atol=1e-10)
    degree = torch.diag(laplacian)
    assert float(degree.min()) >= 2.0 and float(degree.max()) >= 4.0  # corners are linked to their neighbours, the interior to four or more
    off_diagonal = laplacian - torch.diag(degree)
    assert float(off_diagonal.max()) <= 1e-12 and set(np.unique(off_diagonal.numpy().round(8))) <= {0.0, -1.0}


def test_a_region_with_few_observations_borrows_strength_from_its_neighbours():
    train, truth = _regions(rows_per_region=3, seed=1)
    test, _ = _regions(rows_per_region=3, seed=2)
    nominal = ta.StaticTAM("target ~ c(region, n_cat=36, topo='nominal', ap=-1)").fit(train)
    graph = ta.StaticTAM("target ~ c(region, n_cat=36, topo='graph', coords='x:y', k_neighbors=4, ap=-1)").fit(train)

    def error(model):
        return float(np.sqrt(np.mean((model.predict(test)["Estimatedtarget"].to_numpy() - truth[test["region"].to_numpy()]) ** 2)))

    assert error(graph) < 0.9 * error(nominal)


def test_a_code_the_model_never_saw_has_no_effect():
    train, _ = _regions(rows_per_region=4, seed=3)
    model = ta.StaticTAM("target ~ c(region, n_cat=36, topo='graph', coords='x:y', ap=-2)").fit(train)
    probe = pd.DataFrame({"region": [3, 99], "x": [3.0, 9.0], "y": [0.0, 9.0], "target": 0.0})
    parts = model.decompose_prediction(probe)
    assert abs(parts["effect_region"].iloc[0]) > 0.0
    assert parts["effect_region"].iloc[1] == 0.0


def test_it_works_inside_a_tensor_product_and_with_the_codes_of_the_data():
    rng = np.random.default_rng(5)
    frame, _ = _regions(rows_per_region=20, seed=4)
    frame["region"] = frame["region"] * 3 + 7  # codes 7, 10, 13 ...: no n_cat given
    frame["t"] = rng.uniform(0, 1, len(frame))
    frame["target"] = frame["target"] + np.sin(4 * frame["t"])
    model = ta.StaticTAM("target ~ s(t, k=6) + c(region, topo='graph', coords='x:y', ap=-3)").fit(frame)
    assert np.all(np.isfinite(model.predict(frame)["Estimatedtarget"].to_numpy()))
    te = ta.StaticTAM("target ~ te(s(t, k=5), c(region, topo='graph', coords='x:y', ap=-3))").fit(frame)
    assert np.all(np.isfinite(te.predict(frame)["Estimatedtarget"].to_numpy()))


def test_nominal_and_ordinal_are_unchanged():
    frame, _ = _regions(rows_per_region=5)
    a = ta.StaticTAM("target ~ c(region, n_cat=36, topo='nominal', ap=-2)").fit(frame).predict(frame)["Estimatedtarget"].to_numpy()
    b = ta.StaticTAM("target ~ c(region, n_cat=36, ap=-2)").fit(frame).predict(frame)["Estimatedtarget"].to_numpy()
    assert np.array_equal(a, b)
