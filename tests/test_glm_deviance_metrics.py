# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Scoring rules of the non-Gaussian families: unit deviance, deviance explained (D2), and the deviance names accepted by the
model-selection code of AutoTAM.
"""
import numpy as np
import pytest
import torch

from tam.evaluation import calculate_regression_metrics, deviance_explained, deviance_metric, unit_deviance
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.pipeline.ensemble_selector import EnsembleSelector
from tam.model.autotam.pipeline.expert_expander import ExpertExpander
from tam.model.statistics.estimation import build_strategy

FAMILIES = ["gaussian", "poisson", "gamma", "binomial", "negative_binomial", "tweedie"]


def _sample(family, n=400, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.3, 6.0, n)
    if family == "binomial":
        mu = rng.uniform(0.05, 0.95, n)
        return rng.binomial(1, mu).astype(float), mu
    if family == "poisson":
        return rng.poisson(mu).astype(float), mu
    if family == "negative_binomial":
        return rng.negative_binomial(4, 4 / (4 + mu)).astype(float), mu
    if family == "tweedie":
        return np.where(rng.random(n) < 0.2, 0.0, rng.gamma(2.0, mu / 2.0)), mu
    if family == "gamma":
        return rng.gamma(3.0, mu / 3.0), mu
    return mu + rng.normal(size=n), mu


def test_gaussian_matches_squared_error_and_r2():
    y, mu = _sample("gaussian")
    assert np.allclose(unit_deviance(y, mu, "gaussian"), (y - mu) ** 2, atol=1e-12, rtol=0)
    r2 = calculate_regression_metrics(y, mu)["R2"]
    assert deviance_explained(y, mu, "gaussian") == pytest.approx(r2, abs=1e-12)
    out = calculate_regression_metrics(y, mu, family="gaussian")
    assert out["D2"] == pytest.approx(out["R2"], abs=1e-12)
    assert out["Deviance"] == pytest.approx(out["RMSE"] ** 2, abs=1e-12)


def test_default_metrics_keep_their_keys():
    y, mu = _sample("gaussian")
    assert set(calculate_regression_metrics(y, mu)) == {"RMSE", "MAE", "NMAE", "SMAPE", "R2", "MAPE"}


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_at_truth_and_non_negative(family):
    y, mu = _sample(family)
    assert np.all(unit_deviance(y, y if family != "binomial" else np.clip(y, 1e-9, 1 - 1e-9), family) < 1e-5)  # a zero is clipped to 1e-12: 4e-6 for Tweedie
    assert np.all(unit_deviance(y, mu, family) >= -1e-12)
    assert deviance_explained(y, np.clip(y, 1e-9, 1 - 1e-9) if family == "binomial" else y, family) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("family", ["poisson", "gamma", "binomial", "negative_binomial", "tweedie"])
def test_agrees_with_the_unit_deviance_used_by_the_fit(family):
    y, mu = _sample(family)
    kwargs = {"negative_binomial": dict(dispersion=0.1), "tweedie": dict(power=1.5)}.get(family, {})
    strategy = build_strategy(family, **kwargs)
    reference = strategy.unit_deviance(torch.as_tensor(y), torch.as_tensor(mu)).numpy()
    assert np.allclose(unit_deviance(y, mu, family, power=1.5, dispersion=0.1), reference, rtol=1e-6, atol=1e-6)


def test_proper_scoring_rule_on_gamma():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 2, 3000)
    mu = np.exp(2 * x)
    y = rng.gamma(2.0, mu / 2.0)
    wrong = np.exp(2 * x) * (1.0 + 0.5 * np.sin(5 * x))
    assert unit_deviance(y, mu, "gamma").mean() < unit_deviance(y, wrong, "gamma").mean()
    assert deviance_explained(y, mu, "gamma") > deviance_explained(y, wrong, "gamma")


def test_negative_prediction_is_heavily_penalised_not_nan():
    y = np.array([3.0, 1.0, 2.0, 5.0])
    good = np.array([2.5, 1.0, 2.0, 5.0])
    bad = np.array([-3.0, 1.0, 2.0, 5.0])
    assert np.all(np.isfinite(unit_deviance(y, bad, "poisson")))
    assert unit_deviance(y, bad, "poisson").sum() > 10 * unit_deviance(y, good, "poisson").sum()


def test_logloss_is_half_the_binomial_deviance():
    y, mu = _sample("binomial")
    assert deviance_metric("logloss", y, mu) == pytest.approx(deviance_metric("binomial_deviance", y, mu) / 2.0)
    assert deviance_metric("rmse", y, mu) is None


def test_unknown_family_and_bad_parameters_raise():
    with pytest.raises(ValueError):
        unit_deviance(np.ones(3), np.ones(3), "lognormal")
    with pytest.raises(ValueError):
        unit_deviance(np.ones(3), np.ones(3), "tweedie", power=2.5)


@pytest.mark.parametrize("holder", [DragTAM, EnsembleSelector, ExpertExpander])
def test_model_selection_accepts_the_deviance_names(holder):
    y, mu = _sample("poisson")
    scorer = holder.__new__(holder)
    assert scorer._calculate_error(y, mu, "poisson_deviance") == pytest.approx(float(np.mean(unit_deviance(y, mu, "poisson"))))
    assert scorer._calculate_error(y, mu, "rmse") == pytest.approx(float(np.sqrt(np.mean((y - mu) ** 2))))


def test_selection_by_deviance_prefers_the_positive_expert():
    # A Gaussian expert that fits the squares well but predicts negative counts loses on the Poisson deviance.
    rng = np.random.default_rng(5)
    mu = np.exp(rng.uniform(-3, 2, 3000))
    y = rng.poisson(mu).astype(float)
    family_expert = mu
    gaussian_expert = mu + rng.normal(0, 0.15, mu.size)
    assert deviance_metric("poisson_deviance", y, family_expert) < deviance_metric("poisson_deviance", y, gaussian_expert)
