# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Tests for the reweighted penalized least-squares extension (expectile / GLM / robust losses).

Covers the invariants that keep the addition safe and correct:
* the per-sample weighting primitive is exact and its ``None`` path is bit-identical to the legacy solve;
* ``loss='l2'`` is unchanged and ``expectile(tau=0.5)`` reduces to it;
* expectile levels are monotone in ``tau``; GLM (gamma) fits are positive and correlated;
* robust losses down-weight gross outliers; results are reproducible.
"""
import numpy as np
import pandas as pd
import torch

from tam import StaticTAM
from tam.common.utils import _ensure_dummies, _balance_groups
from tam.model._math import _compute_weighted_covariances
from tam.model.statistics.estimation import build_strategy


def _prepared_tensors(model, dataframe):
    dummied = _ensure_dummies(dataframe.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    return model._prepare_data(balanced, target_col=model.target_col_)


def test_sample_weights_none_is_identity_and_ones_match():
    torch.manual_seed(0)
    phi = torch.randn(2, 40, 3, dtype=torch.float64)
    y = torch.randn(2, 40, 1, dtype=torch.float64)
    loss = torch.eye(1, dtype=torch.float64)
    cov_x_none, cov_xy_none = _compute_weighted_covariances(phi, y, loss, None)
    cov_x_one, cov_xy_one = _compute_weighted_covariances(phi, y, loss, torch.ones(2, 40, 1, dtype=torch.float64))
    assert torch.allclose(cov_x_none.cpu(), (phi.mT @ phi).cpu(), atol=1e-10)
    assert torch.allclose(cov_xy_none.cpu(), (phi.mT @ y).cpu(), atol=1e-10)
    assert torch.allclose(cov_x_none.cpu(), cov_x_one.cpu(), atol=1e-12)
    assert torch.allclose(cov_xy_none.cpu(), cov_xy_one.cpu(), atol=1e-12)


def test_weighted_covariance_matches_manual_wls():
    torch.manual_seed(1)
    phi = torch.randn(2, 60, 4, dtype=torch.float64)
    y = torch.randn(2, 60, 1, dtype=torch.float64)
    weights = torch.rand(2, 60, 1, dtype=torch.float64) + 0.1
    diag_w = torch.diag_embed(weights.squeeze(-1))
    cov_x, cov_xy = _compute_weighted_covariances(phi, y, torch.eye(1, dtype=torch.float64), weights)
    assert torch.allclose(cov_x.cpu(), (phi.mT @ diag_w @ phi).cpu(), atol=1e-9)
    assert torch.allclose(cov_xy.cpu(), (phi.mT @ diag_w @ y).cpu(), atol=1e-9)


def test_pwls_step_weighted_matches_manual():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)
    frame = pd.DataFrame({"x": x, "y": 2.0 * x + 0.5 + 0.1 * rng.standard_normal(300)})
    model = StaticTAM("y ~ s(x)").fit(frame)
    x_tensor, y_tensor, _ = _prepared_tensors(model, frame)
    penalty = model._build_penalty_matrix().to(torch.float64)
    n_samples = x_tensor.shape[1]
    weights = torch.rand(x_tensor.shape[0], n_samples, 1, dtype=torch.float64) + 0.1

    coeffs = model._solve_pwls_step(x_tensor, y_tensor, weights)
    design = model._build_design_matrix(x_tensor.to(torch.float64))
    diag_w = torch.diag_embed(weights.squeeze(-1))
    lhs = design.mT @ diag_w @ design + n_samples * penalty + 1e-6 * n_samples * torch.eye(design.shape[-1], dtype=torch.float64)
    coeffs_manual = torch.linalg.solve(lhs, design.mT @ diag_w @ y_tensor.to(torch.float64))
    assert torch.allclose(coeffs.cpu().to(torch.float64), coeffs_manual.cpu(), atol=1e-7)
    # None path reproduces the ordinary fit exactly
    assert torch.allclose(model._solve_pwls_step(x_tensor, y_tensor, None).cpu().to(torch.float64),
                          model.coefficients_.cpu().to(torch.float64), atol=1e-9)


def _fit_predict(loss, frame, **kwargs):
    return StaticTAM("y ~ s(x)", loss=loss, **kwargs).fit(frame).predict(frame)["Estimatedy"].to_numpy()


def test_l2_default_matches_explicit():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(400)
    frame = pd.DataFrame({"x": x, "y": 1.5 * x + 0.3 * rng.standard_normal(400)})
    default = StaticTAM("y ~ s(x)").fit(frame).predict(frame)["Estimatedy"].to_numpy()
    explicit = _fit_predict("l2", frame)
    assert np.allclose(default, explicit, atol=1e-9)


def test_expectile_half_equals_l2():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(400)
    frame = pd.DataFrame({"x": x, "y": 1.5 * x + 0.3 * rng.standard_normal(400)})
    assert np.allclose(_fit_predict("l2", frame), _fit_predict("expectile", frame, loss_kwargs={"tau": 0.5}), atol=1e-4)


def test_expectile_levels_are_monotone_in_tau():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(600)
    frame = pd.DataFrame({"x": x, "y": 2.0 * x + 0.4 * rng.standard_normal(600)})
    fractions = {tau: float(np.mean(frame["y"].to_numpy() <= _fit_predict("expectile", frame, loss_kwargs={"tau": tau})))
                 for tau in (0.1, 0.5, 0.9)}
    assert fractions[0.1] < fractions[0.5] < fractions[0.9]


def test_gamma_predicts_positive_and_correlated():
    rng = np.random.default_rng(6)
    x = rng.random(500)
    frame = pd.DataFrame({"x": x, "y": np.exp(1.0 + 1.5 * x) * rng.gamma(8.0, 1.0 / 8.0, 500)})
    predicted = _fit_predict("gamma", frame)
    assert np.all(predicted > 0)
    assert np.corrcoef(predicted, frame["y"].to_numpy())[0, 1] > 0.6


def test_student_t_is_robust_to_outliers():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    y = 1.5 * x + 0.2 * rng.standard_normal(500)
    y[:40] += 40.0  # gross outliers
    frame = pd.DataFrame({"x": x, "y": y})
    clean = slice(40, None)
    rmse = lambda pred: float(np.sqrt(np.mean((pred[clean] - 1.5 * x[clean]) ** 2)))
    assert rmse(_fit_predict("student_t", frame, loss_kwargs={"nu": 3.0})) < rmse(_fit_predict("l2", frame))


def test_results_are_reproducible():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(300)
    frame = pd.DataFrame({"x": x, "y": np.exp(0.5 + x) * rng.gamma(6.0, 1.0 / 6.0, 300)})
    assert np.allclose(_fit_predict("gamma", frame), _fit_predict("gamma", frame), atol=1e-10)


def test_poisson_predicts_positive_rate():
    rng = np.random.default_rng(9)
    x = rng.uniform(0.0, 1.0, 600)
    rate = np.exp(0.5 + 1.5 * x)
    frame = pd.DataFrame({"x": x, "y": rng.poisson(rate).astype(float)})
    predicted = _fit_predict("poisson", frame)
    assert np.all(predicted > 0)
    assert np.corrcoef(predicted, rate)[0, 1] > 0.8


def test_binomial_predicts_calibrated_probabilities():
    rng = np.random.default_rng(10)
    x = rng.uniform(-2.0, 2.0, 800)
    probability = 1.0 / (1.0 + np.exp(-2.0 * x))
    outcome = (rng.uniform(size=800) < probability).astype(float)
    frame = pd.DataFrame({"x": x, "y": outcome})
    predicted = _fit_predict("binomial", frame)
    assert np.all((predicted >= 0.0) & (predicted <= 1.0))
    assert predicted[outcome == 1].mean() > predicted[outcome == 0].mean()


def test_huber_is_robust_to_outliers():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(500)
    y = 1.5 * x + 0.2 * rng.standard_normal(500)
    y[:40] += 40.0
    frame = pd.DataFrame({"x": x, "y": y})
    clean = slice(40, None)
    rmse = lambda pred: float(np.sqrt(np.mean((pred[clean] - 1.5 * x[clean]) ** 2)))
    assert rmse(_fit_predict("huber", frame, loss_kwargs={"delta": 1.345})) < rmse(_fit_predict("l2", frame))


def test_build_strategy_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        build_strategy("not_a_loss")


def test_glm_predict_on_grouped_dataset():
    # Two groups with different Poisson rates; predict() must map back to the mean scale
    # (positive) per group when group_col/date_col are set. Guards the predict override.
    rng = np.random.default_rng(21)
    frames = []
    for g, intercept in enumerate((1.0, 2.5)):
        x = rng.uniform(-1.0, 1.0, 400)
        rate = np.exp(intercept + 1.2 * x)
        frames.append(pd.DataFrame({"gid": g, "t": np.arange(400), "x": x,
                                    "y": rng.poisson(rate).astype(float)}))
    frame = pd.concat(frames, ignore_index=True)
    model = StaticTAM("y ~ s(x)", group_col="gid", date_col="t", loss="poisson").fit(frame)
    predicted = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(predicted)) and np.all(predicted > 0)
    # Group 1 has the higher intercept, so its mean prediction must exceed group 0's.
    g_id = frame["gid"].to_numpy()
    assert predicted[g_id == 1].mean() > predicted[g_id == 0].mean()
