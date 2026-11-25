# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The REML search of ``auto_fit(criterion='reml')``: a short fixed-point warm start and a safeguarded Newton search with the analytic Hessian.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

import tam as ta
from tam.common.utils import _balance_groups, _ensure_dummies
from tam.model._math import _compute_weighted_covariances
from tam.model._reml import PenaltyBasis, reml_objective

REASONS = ("newton_ok", "newton_bounds", "newton_step_limit")


def _curve(loss="l2", n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    eta = np.sin(1.5 * x)
    if loss == "poisson":
        y = rng.poisson(np.exp(0.5 * eta)).astype(float)
    elif loss == "binomial":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * eta))).astype(float)
    else:
        y = eta + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.uniform(-1, 1, n), "y": y})


def _reference_optimum(formula, frame, known_scale=None):
    """The optimum of the criterion by an independent quasi-Newton search from a grid of starts."""
    model = ta.StaticTAM(formula)
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    x, y, _ = model._prepare_data(balanced, target_col=model.target_col_)
    phi = model._build_design_matrix(x).double()
    cov_x, cov_xy = _compute_weighted_covariances(phi, y.double(), torch.eye(1, dtype=torch.float64))
    y_sq = (y.double() ** 2).sum(dim=(1, 2))
    basis = PenaltyBasis(model.effects_list_)
    best = np.inf
    for start in (-4.0, 0.0, 4.0):
        result = minimize(lambda r: reml_objective(r, cov_x, cov_xy, y_sq, basis, x.shape[1], known_scale), np.full(basis.n_coordinates, start),
                          jac=True, method="L-BFGS-B", bounds=[(-69.0, 14.0)] * basis.n_coordinates)
        best = min(best, float(result.fun))
    return -best


def test_the_search_reaches_the_optimum_of_an_independent_quasi_newton_search():
    frame = _curve()
    formula = "y ~ s(x, k=12) + s(z, k=8)"
    fitted = ta.StaticTAM(formula).auto_fit(frame, criterion="reml")
    assert fitted.reml_score_ >= _reference_optimum(formula, frame) - 0.05
    assert fitted.reml_stop_reason_ in REASONS
    assert len(fitted.reml_iterations_) == 1 and fitted.reml_iterations_[0] > 0


def test_the_search_is_deterministic():
    frame = _curve()
    first = ta.StaticTAM("y ~ s(x, k=14) + s(z, k=14)").auto_fit(frame, criterion="reml")
    second = ta.StaticTAM("y ~ s(x, k=14) + s(z, k=14)").auto_fit(frame, criterion="reml")
    assert first.reml_score_ == second.reml_score_ and first.reml_stop_reason_ == second.reml_stop_reason_


@pytest.mark.parametrize("loss", ["l2", "poisson", "gamma", "binomial"])
def test_a_single_admissible_value_is_the_plain_fit(loss):
    frame = _curve("binomial" if loss == "binomial" else ("poisson" if loss == "poisson" else "l2"))
    if loss == "gamma":
        frame["y"] = np.exp(0.3 * np.sin(1.5 * frame["x"])) * np.random.default_rng(3).gamma(4.0, 0.25, len(frame))
    tuned = ta.StaticTAM("y ~ s(x, k=12, ap=-3)", loss=loss).auto_fit(frame, criterion="reml", alpha_p_list=[-3.0])
    plain = ta.StaticTAM("y ~ s(x, k=12, ap=-3)", default_alpha_p=-3.0, loss=loss).fit(frame)
    assert torch.allclose(tuned.coefficients_.cpu(), plain.coefficients_.cpu(), atol=1e-8, rtol=0)


@pytest.mark.parametrize("loss", ["poisson", "binomial"])
def test_a_glm_fit_predicts_like_the_squared_loss_reference(loss):
    frame = _curve(loss)
    model = ta.StaticTAM("y ~ s(x, k=12) + s(z, k=8)", loss=loss).auto_fit(frame, criterion="reml")
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    truth = np.exp(0.5 * np.sin(1.5 * frame["x"])) if loss == "poisson" else 1 / (1 + np.exp(-1.5 * np.sin(1.5 * frame["x"])))
    assert np.all(np.isfinite(pred)) and np.sqrt(np.mean((pred - truth) ** 2)) < 0.25


def test_a_tiny_sample_with_a_wide_design_has_a_deterministic_float64_score():
    frame = _curve(n=20, seed=5)
    scores = [ta.StaticTAM("y ~ s(x, k=30) + s(z, k=30)").auto_fit(frame, criterion="reml").reml_score_ for _ in range(2)]
    assert scores[0] == scores[1] and np.isfinite(scores[0])


@pytest.mark.parametrize("truth", ["noise", "linear", "collinear"])
def test_flat_and_bound_coordinates_do_not_break_the_search(truth):
    rng = np.random.default_rng(2)
    n = 300
    x, z = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    if truth == "noise":
        y = rng.normal(size=n)
    elif truth == "linear":
        y = 1.0 + 2.0 * x + 0.1 * rng.normal(size=n)
    else:
        z = x + 1e-6 * rng.normal(size=n)
        y = np.sin(2 * x) + 0.2 * rng.normal(size=n)
    model = ta.StaticTAM("y ~ s(x, k=10) + s(z, k=10)").auto_fit(pd.DataFrame({"x": x, "z": z, "y": y}), criterion="reml")
    assert model.reml_stop_reason_ in ("newton_ok", "newton_bounds")
    assert np.isfinite(model.reml_score_)


def test_a_quasi_separated_binomial_fit_stays_finite():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 200)
    frame = pd.DataFrame({"x": x, "y": (x + 0.02 * rng.normal(size=200) > 0).astype(float)})
    model = ta.StaticTAM("y ~ s(x, k=10)", loss="binomial").auto_fit(frame, criterion="reml")
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(pred)) and np.all((pred >= 0) & (pred <= 1)) and np.isfinite(model.reml_score_)


def test_a_zero_inflated_tweedie_fit_stays_finite():
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, 400)
    y = np.where(rng.uniform(size=400) < 0.45, 0.0, rng.gamma(2.0, np.exp(0.5 + 0.8 * x) / 2.0))
    model = ta.StaticTAM("y ~ s(x, k=10)", loss="tweedie", loss_kwargs={"power": 1.5}).auto_fit(pd.DataFrame({"x": x, "y": y}), criterion="reml")
    assert np.all(np.isfinite(model.predict(pd.DataFrame({"x": x}))["Estimatedy"].to_numpy())) and np.isfinite(model.reml_score_)
