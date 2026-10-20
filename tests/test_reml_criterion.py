# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``auto_fit(criterion='reml')``: the smoothing parameters maximise the Laplace-approximate restricted marginal likelihood
``-1/2 log|X'WX + S| + 1/2 log|S|_+ - (n - M0)/2 log(sigma^2)`` instead of the generalized cross-validation score.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.common.utils import _balance_groups, _ensure_dummies
from tam.model._reml import PenaltyBasis, reml_objective


def _prepared(model, frame):
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    x, y, _ = model._prepare_data(balanced, target_col=model.target_col_)
    return x, y


def _sine(n=400, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.uniform(0, 1, n), "y": np.sin(2 * np.pi * x) + noise * rng.normal(size=n)})


def test_the_analytic_gradient_matches_finite_differences():
    frame = _sine()
    model = ta.StaticTAM("y ~ s(x, k=12) + s(z, k=8)")
    x, y = _prepared(model, frame)
    basis = PenaltyBasis(model.effects_list_)
    phi = model._build_design_matrix(x).double()
    from tam.model._math import _compute_weighted_covariances
    cov_x, cov_xy = _compute_weighted_covariances(phi, y.double(), torch.eye(1, dtype=torch.float64))
    y_sq = (y.double() ** 2).sum(dim=(1, 2))
    rho = np.array([-1.0 * np.log(10), 1.5, 0.3, 0.7][: basis.n_coordinates])
    value, gradient = reml_objective(rho, cov_x, cov_xy, y_sq, basis, n_samples=x.shape[1])
    numeric = np.zeros_like(rho)
    for i in range(len(rho)):
        step = np.zeros_like(rho)
        step[i] = 1e-4
        numeric[i] = (reml_objective(rho + step, cov_x, cov_xy, y_sq, basis, x.shape[1])[0]
                      - reml_objective(rho - step, cov_x, cov_xy, y_sq, basis, x.shape[1])[0]) / 2e-4
    assert np.allclose(gradient, numeric, rtol=1e-5, atol=1e-5)


def test_the_penalty_basis_reproduces_the_penalty_of_the_effects():
    model = ta.StaticTAM("y ~ s(x, k=10, m_pen=3) + te(s(x, k=5), s(z, k=5)) + s(z, k=6)")
    x, y = _prepared(model, _sine())
    basis = PenaltyBasis(model.effects_list_)
    rho = np.log(np.array([0.2 * (i + 1) for i in range(basis.n_coordinates)]))
    rebuilt = basis.penalty(rho)
    position = 0
    for effect in model.effects_list_:
        count = effect.n_penalty_coordinates
        effect.set_penalty_coordinates([float(np.exp(v)) for v in rho[position:position + count]])
        position += count
    from tam.model.spectrum import build_penalty_from_effects
    assert torch.allclose(rebuilt, build_penalty_from_effects(model.effects_list_), atol=1e-12, rtol=0)


def test_polynomial_signal_is_fitted_exactly_without_numerical_trouble():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 300)
    frame = pd.DataFrame({"x": x, "y": 1.0 + 2.0 * x + 0.05 * rng.normal(size=300)})
    model = ta.StaticTAM("y ~ s(x, k=10)").auto_fit(frame, criterion="reml")
    assert np.all(np.isfinite([e.lambda_p for e in model.effects_list_]))
    assert np.isfinite(model.reml_score_) and model.smoothing_criterion_ == "reml"
    assert np.sqrt(np.mean((model.predict(frame)["Estimatedy"].to_numpy() - (1 + 2 * x)) ** 2)) < 0.02


def test_reml_is_more_stable_than_gcv_on_autocorrelated_noise():
    def lambdas(criterion, seed):
        rng = np.random.default_rng(seed)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.7 * noise[t - 1] + 0.2 * rng.normal()
        frame = pd.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + noise})
        model = ta.StaticTAM("y ~ s(x, k=25)").auto_fit(frame, criterion=criterion)
        truth = np.sin(2 * np.pi * x)
        return float(np.sqrt(np.mean((model.predict(frame)["Estimatedy"].to_numpy() - truth) ** 2)))

    gcv = np.mean([lambdas("gcv", s) for s in range(12)])
    reml = np.mean([lambdas("reml", s) for s in range(12)])
    assert reml < gcv


def test_default_criterion_is_gcv_and_unchanged():
    frame = _sine()
    default = ta.StaticTAM("y ~ s(x, k=10)").auto_fit(frame)
    explicit = ta.StaticTAM("y ~ s(x, k=10)").auto_fit(frame, criterion="gcv")
    assert np.array_equal(default.predict(frame)["Estimatedy"].to_numpy(), explicit.predict(frame)["Estimatedy"].to_numpy())
    assert default.smoothing_criterion_ == "gcv" and default.reml_score_ is None


def test_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="criterion"):
        ta.StaticTAM("y ~ s(x, k=10)").auto_fit(_sine(), criterion="aic")


@pytest.mark.parametrize("loss", ["poisson", "binomial", "gamma"])
def test_reml_runs_on_the_glm_losses(loss):
    rng = np.random.default_rng(3)
    n = 1200
    x = rng.uniform(0, 1, n)
    eta = 0.5 + np.sin(3 * x)
    if loss == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif loss == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.gamma(4.0, np.exp(eta) / 4.0)
    frame = pd.DataFrame({"x": x, "y": y})
    model = ta.StaticTAM("y ~ s(x, k=12)", loss=loss).auto_fit(frame, criterion="reml")
    pred = model.predict(frame)["Estimatedy"].to_numpy()
    assert np.all(np.isfinite(pred)) and np.all(pred > 0) and model.smoothing_criterion_ == "reml"
    assert np.corrcoef(pred, (np.exp(eta) if loss != "binomial" else 1 / (1 + np.exp(-eta))))[0, 1] > 0.8


def test_reml_handles_per_level_and_adaptive_coordinates():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.uniform(0, 1, 2 * n)
    g = np.repeat([0, 1], n)
    frame = pd.DataFrame({"x": x, "g": g, "y": np.where(g == 0, x, np.sin(12 * x)) + np.where(g == 0, 0.5, 0.05) * rng.normal(size=2 * n)})
    model = ta.StaticTAM("y ~ c(g, n_cat=2, ap=-6) + te(s(x, k=12), c(g, n_cat=2), mode='per_level')").auto_fit(frame, criterion="reml")
    lam = next(e for e in model.effects_list_ if e.__class__.__name__ == "TensorProductEffect").penalty_coordinates()
    assert lam[0] > 5.0 * lam[1]
    adaptive = ta.StaticTAM("y ~ s(x, k=20, m_pen=3)").auto_fit(frame[frame["g"] == 1], criterion="reml")
    assert np.isfinite(adaptive.reml_score_)


def test_reml_with_an_offset_and_with_several_groups():
    rng = np.random.default_rng(5)
    n = 600
    exposure = rng.uniform(1, 30, n)
    x = rng.uniform(0, 1, n)
    frame = pd.DataFrame({"x": x, "log_e": np.log(exposure), "y": rng.poisson(exposure * np.exp(0.3 + np.sin(3 * x))).astype(float)})
    model = ta.StaticTAM("y ~ offset(log_e) + s(x, k=10)", loss="poisson").auto_fit(frame, criterion="reml")
    low = pd.DataFrame({"x": [0.5, 0.5], "log_e": [1.0, 1.0 + np.log(2.0)], "y": 0.0})
    pred = model.predict(low)["Estimatedy"].to_numpy()
    assert pred[1] / pred[0] == pytest.approx(2.0, rel=1e-9)

    frames = []
    for g in range(3):
        xg = rng.uniform(0, 1, 300)
        frames.append(pd.DataFrame({"g": g, "date": np.arange(300), "x": xg, "y": np.sin(4 * xg) + 0.2 * rng.normal(size=300)}))
    panel = pd.concat(frames, ignore_index=True)
    grouped = ta.StaticTAM("y ~ s(x, k=10)", group_col="g", date_col="date").auto_fit(panel, criterion="reml")
    assert np.all(np.isfinite(grouped.predict(panel)["Estimatedy"].to_numpy())) and np.isfinite(grouped.reml_score_)
