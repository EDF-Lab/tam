# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``auto_fit`` (generalized cross-validation) and ``grid_search_fit`` on the non-Gaussian losses.

Both used to ignore the loss: GCV ran on the raw response with an identity link, and the coefficients were then passed through the
inverse link by ``predict``. They now work on the working response and the Fisher weights of the loss, and the returned model keeps
the loss.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.common.utils import _ensure_dummies, _balance_groups
from tam.model._dispatcher_gcv import smart_solve_gcv


def _curve(loss, n=1200, seed=0, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    mean = np.exp(1.2 + 1.1 * np.sin(3.0 * x))
    if loss == "poisson":
        y = rng.poisson(mean).astype(float)
    elif loss == "gamma":
        y = rng.gamma(4.0, mean / 4.0)
    elif loss == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * np.sin(3.0 * x)))).astype(float)
    elif loss == "tweedie":
        y = np.where(rng.random(n) < 0.3, 0.0, rng.gamma(2.0, mean / 2.0))
    else:  # negative_binomial
        y = rng.negative_binomial(3, 3.0 / (3.0 + mean)).astype(float)
    return pd.DataFrame({"x": x, "y": y})


def _deviance(loss, y, mu):
    mu = np.clip(mu, 1e-9, None)
    y = np.asarray(y, dtype=float)
    if loss == "poisson" or loss == "negative_binomial":
        return float(np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))))
    if loss == "gamma":
        return float(np.mean(2 * (-np.log(y / mu) + (y - mu) / mu)))
    if loss == "binomial":
        mu = np.clip(mu, 1e-9, 1 - 1e-9)
        return float(np.mean(-2 * (y * np.log(mu) + (1 - y) * np.log(1 - mu))))
    return float(np.mean((y - mu) ** 2))  # tweedie: squared error is enough to rank the two fits here


@pytest.mark.parametrize("loss", ["poisson", "gamma", "binomial", "tweedie", "negative_binomial"])
def test_auto_fit_predicts_on_the_response_scale(loss):
    frame = _curve(loss)
    model = ta.StaticTAM("y ~ s(x, k=20)", loss=loss).auto_fit(frame)
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(pred)) and np.all(pred >= 0)
    if loss == "binomial":
        assert np.all(pred <= 1)
    assert _deviance(loss, frame["y"], pred) < _deviance(loss, frame["y"], np.full(len(frame), frame["y"].mean()))


@pytest.mark.parametrize("loss", ["poisson", "gamma", "binomial"])
def test_auto_fit_with_one_value_is_fit(loss):
    frame = _curve(loss)
    tuned = ta.StaticTAM("y ~ s(x, k=12, ap=-3)", loss=loss).auto_fit(frame, alpha_p_list=[-3.0])
    plain = ta.StaticTAM("y ~ s(x, k=12, ap=-3)", default_alpha_p=-3.0, loss=loss).fit(frame)  # the list sets the intercept too
    assert torch.allclose(tuned.coefficients_.cpu(), plain.coefficients_.cpu(), atol=1e-8, rtol=0)


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_auto_fit_beats_an_untuned_fit_out_of_sample(loss):
    train, test = _curve(loss, n=400, seed=1), _curve(loss, n=4000, seed=2)
    untuned = ta.StaticTAM("y ~ s(x, k=30)", loss=loss).fit(train)
    tuned = ta.StaticTAM("y ~ s(x, k=30)", loss=loss).auto_fit(train)
    dev_untuned = _deviance(loss, test["y"], untuned.predict(test)["Estimatedy"].to_numpy())
    dev_tuned = _deviance(loss, test["y"], tuned.predict(test)["Estimatedy"].to_numpy())
    assert dev_tuned < dev_untuned


def test_gaussian_auto_fit_is_unchanged_by_unit_weights():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 500)
    frame = pd.DataFrame({"x": x, "y": np.sin(3 * x) + 0.3 * rng.normal(size=500)})
    model = ta.StaticTAM("y ~ s(x, k=15)")
    data = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(data, model.group_col_, model.date_col_, method="drop")
    x_t, y_t, _ = model._prepare_data(balanced, target_col=model.target_col_)
    loss = model._build_loss_matrix()
    plain, lam_plain, _ = smart_solve_gcv(x_t, y_t, model.effects_list_, loss, (-12.0, 4.0))
    ones = torch.ones_like(y_t)
    weighted, lam_ones, _ = smart_solve_gcv(x_t, y_t, model.effects_list_, loss, (-12.0, 4.0), sample_weights=ones)
    assert np.allclose(lam_plain, lam_ones)
    assert torch.allclose(plain.cpu(), weighted.cpu(), atol=1e-10, rtol=0)


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
def test_grid_search_fit_scores_on_the_response_scale(loss):
    train, val = _curve(loss, n=600, seed=5), _curve(loss, n=600, seed=6)
    grid = {"pen": [-6.0, -3.0, 0.0]}
    model = ta.StaticTAM("y ~ s(x, k=15, ap='pen')", loss=loss).grid_search_fit(train, val, grid)
    pred = model.predict(val)["Estimatedy"].to_numpy()
    assert np.all(pred > 0)
    reported = model.validation_rmse_
    assert reported == pytest.approx(float(np.sqrt(np.mean((val["y"].to_numpy() - pred) ** 2))), rel=1e-6)
