# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``rbf(x1, others='x2', kernel='tps')``: the thin-plate spline on data-driven centres. Radial basis ``eta(r)`` of the polynomial-constrained
span, penalty ``Z' E Z`` (the bending energy) on the radial part, and the polynomial part of degree ``< m`` (the intercept of the model
is the constant). Isotropic: a rotation of the coordinates does not change the fit.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.spectrum import RBFEffect


def _surface(n, seed, noise=0.1):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    truth = np.sin(2.5 * x1) * np.cos(3.0 * x2) + 0.5 * x1
    return pd.DataFrame({"x1": x1, "x2": x2, "y": truth + noise * rng.normal(size=n)}), truth


def _effect(n_centers=20, **kwargs):
    return RBFEffect("x1", n_centers, None, None, 1e-3, ["x2"], "continue", kernel="tps", **kwargs)  # no clamping: a rotated point may leave the square


def _initialised(effect, n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.uniform(-1, 1, (1, n, 2)))
    effect.initialize(x)
    return effect, x


def test_the_name_of_the_kernel_is_checked():
    with pytest.raises(ValueError, match="kernel"):
        RBFEffect("x1", 20, None, None, 1e-3, ["x2"], "linear", kernel="sinc")


def test_the_order_must_exceed_half_the_dimension():
    with pytest.raises(ValueError, match="2m"):
        RBFEffect("x1", 20, None, None, 1e-3, ["x2", "x3", "x4"], "linear", kernel="tps", m=2)


def test_constraint_and_energy():
    effect, _ = _initialised(_effect())
    centres = effect.centers
    t = torch.cat([torch.ones(len(centres), 1, dtype=centres.dtype), centres], dim=1)  # constant, x1, x2
    z = effect._null_basis
    assert torch.allclose(t.T @ z, torch.zeros(3, z.shape[1], dtype=t.dtype), atol=1e-10)
    assert torch.allclose(z.T @ z, torch.eye(z.shape[1], dtype=z.dtype), atol=1e-10)
    penalty = effect.build_penalty_matrix()
    assert float(torch.linalg.eigvalsh(penalty).min()) > -1e-10
    assert effect.get_n_coeffs() == len(centres) - 1  # the constant is the intercept of the model


def _fit_predictions(effect, x_train, y_train, x_test, lam):
    effect.lambda_p = lam
    phi = effect.transform(x_train)
    ridge = effect.build_penalty_matrix().to(phi.dtype)
    theta = torch.linalg.solve(phi.mT @ phi + x_train.shape[1] * ridge, phi.mT @ y_train)
    return (effect.transform(x_test) @ theta).squeeze(-1)


def test_a_plane_is_reproduced_when_the_penalty_is_huge():
    effect, x = _initialised(_effect(30))
    y = (0.7 * x[..., 0] - 1.3 * x[..., 1]).unsqueeze(-1)
    fit = _fit_predictions(effect, x, y, x, lam=1e6)
    assert float((fit - y.squeeze(-1)).abs().max()) < 1e-3


def test_the_fit_does_not_depend_on_the_orientation_of_the_coordinates():
    effect, x = _initialised(_effect(30))
    y = (torch.sin(2 * x[..., 0]) * torch.cos(3 * x[..., 1])).unsqueeze(-1)
    angle = 0.9
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=x.dtype)
    rotated = _effect(30)
    rotated.initialize(x @ rotation.T)
    rotated.centers = effect.centers @ rotation.T  # the same centres, seen in the rotated frame
    rotated._prepare_basis()
    a = _fit_predictions(effect, x, y, x, lam=1e-2)
    b = _fit_predictions(rotated, x @ rotation.T, y, x @ rotation.T, lam=1e-2)
    assert torch.allclose(a, b, atol=1e-6)


def test_the_surface_is_recovered_through_a_formula():
    train, _ = _surface(1500, 1)
    test, truth = _surface(4000, 2, noise=0.0)
    model = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=60, kernel='tps', ap=-4)").fit(train)
    rmse = float(np.sqrt(np.mean((model.predict(test)["Estimatedy"].to_numpy() - truth) ** 2)))
    assert rmse < 0.05


def test_the_null_space_weight_is_a_second_coordinate():
    effect, _ = _initialised(_effect(), 400)
    assert effect.n_penalty_coordinates == 1
    with_null = RBFEffect("x1", 20, None, None, 1e-3, ["x2"], "linear", kernel="tps", lambda_null=1e-2)
    with_null.initialize(torch.tensor(np.random.default_rng(0).uniform(-1, 1, (1, 400, 2))))
    assert with_null.n_penalty_coordinates == 2
    with_null.set_penalty_coordinates([0.5, 0.25])
    assert with_null.penalty_coordinates() == [0.5, 0.25]
    penalty = with_null.build_penalty_matrix()
    assert float(penalty[-1, -1]) == pytest.approx(0.25)  # the polynomial block is last


def test_auto_fit_tunes_both_coordinates():
    train, _ = _surface(1000, 3)
    model = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=40, kernel='tps', ap=-3, ap_null=-2)").auto_fit(train)
    spline = next(e for e in model.effects_list_ if isinstance(e, RBFEffect))
    assert spline.n_penalty_coordinates == 2


def test_far_outside_the_data_the_prediction_stays_finite_and_moderate():
    train, _ = _surface(800, 4)
    model = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=40, kernel='tps', ap=-3)").fit(train)
    far = pd.DataFrame({"x1": [5.0, -4.0], "x2": [6.0, -3.0], "y": 0.0})
    prediction = model.predict(far)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(prediction)) and np.all(np.abs(prediction) < 50.0)


def test_the_other_kernels_are_unchanged():
    train, _ = _surface(600, 5)
    a = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=30, ap=-3)").fit(train).predict(train)["Estimatedy"].to_numpy()
    b = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=30, kernel='matern', ap=-3)").fit(train).predict(train)["Estimatedy"].to_numpy()
    assert np.array_equal(a, b)


def test_grid_search_uses_the_thin_plate_penalty_and_survives_few_distinct_points():
    rng = np.random.default_rng(7)
    base = rng.uniform(0, 1, (35, 2))
    rows = base[rng.integers(0, 35, 400)]  # 400 rows on 35 distinct points, fewer than the 50 centres asked
    frame = pd.DataFrame({"x1": rows[:, 0], "x2": rows[:, 1]})
    frame["y"] = np.sin(3 * frame["x1"]) + frame["x2"] + 0.1 * rng.normal(size=len(frame))
    train, val = frame.iloc[:300], frame.iloc[300:]
    model = ta.StaticTAM("y ~ rbf(x1, others='x2', n_centers=50, kernel='tps', ap='pen')").grid_search_fit(train, val, {"pen": [-6.0, -2.0]})
    effect = next(e for e in model.effects_list_ if isinstance(e, RBFEffect))
    assert effect.n_centers <= 35
    penalty = effect.build_penalty_matrix()
    assert penalty.shape[0] == effect.get_n_coeffs() and not torch.allclose(penalty, torch.diag(torch.diagonal(penalty)))  # a bending energy, not a ridge


def test_collinear_centres_are_refused():
    effect = RBFEffect("x1", 12, None, None, 1e-3, ["x2"], "continue", kernel="tps")
    line = torch.linspace(-1, 1, 30, dtype=torch.float64)
    with pytest.raises(ValueError, match="collinear|rank"):
        effect.initialize(torch.stack([line, 2 * line], dim=-1).unsqueeze(0))
