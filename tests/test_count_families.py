# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Negative binomial and Tweedie families of :class:`StaticTAM` (``loss="negative_binomial"`` and ``loss="tweedie"``, log link), and
``predict_quantiles`` for the count and zero-inflated-positive models: discrete quantiles for Poisson and negative binomial, the
compound Poisson-gamma law for Tweedie. Quantiles are non-negative by construction.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import tam as ta
from tam.model.statistics.estimation import build_strategy

N = 20000


def _frame(y, x):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(y), freq="h"), "x": x, "y": y})


def _negative_binomial(seed=0, n=N, theta=2.0, intercept=4.0, slope=0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = np.exp(intercept + slope * x)
    return _frame(rng.negative_binomial(theta, theta / (theta + mu)), x), mu


def _tweedie(seed=0, n=N, power=1.5, phi=2.0, intercept=-3.0, slope=1.0):
    """Compound Poisson-gamma draws with mean mu = exp(intercept + slope x) and variance phi mu^power."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = np.exp(intercept + slope * x)
    lam = mu ** (2 - power) / (phi * (2 - power))
    shape = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu ** (power - 1)
    counts = rng.poisson(lam)
    y = np.array([rng.gamma(shape * c, s) if c else 0.0 for c, s in zip(counts, scale)])
    return _frame(y, x), mu


def _model(loss, **loss_kwargs):
    return ta.StaticTAM("y ~ l(x)", date_col="date", loss=loss, loss_kwargs=loss_kwargs or None)


def test_negative_binomial_recovers_the_dispersion():
    df, _ = _negative_binomial()
    model = _model("negative_binomial").fit(df)
    assert abs(model.dispersion_ - 0.5) / 0.5 < 0.10          # the dispersion alpha = 1 / theta
    coef_ratio = model.predict(df)["Estimatedy"].mean() / df["y"].mean()
    assert abs(coef_ratio - 1.0) < 0.02


def test_negative_binomial_interval_covers():
    train, _ = _negative_binomial(seed=0)
    test, _ = _negative_binomial(seed=1)
    model = _model("negative_binomial").fit(train)
    quantiles = model.predict_quantiles(test, (0.1, 0.5, 0.9))
    y = test["y"].to_numpy()
    inside = (y >= quantiles["q0.1"].to_numpy()) & (y <= quantiles["q0.9"].to_numpy())
    assert abs(float(inside.mean()) - 0.8) < 0.03


def test_a_given_dispersion_is_kept_and_not_estimated():
    df, _ = _negative_binomial()
    model = _model("negative_binomial", dispersion=0.3).fit(df)
    assert model.dispersion_ == 0.3


def test_poisson_quantiles_are_the_discrete_poisson_quantiles():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    df = _frame(rng.poisson(np.exp(2.0 + 0.4 * x)), x)
    model = _model("poisson").fit(df)
    mean = model.predict(df)["Estimatedy"].to_numpy()
    quantiles = model.predict_quantiles(df, (0.1, 0.5, 0.9))
    for tau in (0.1, 0.5, 0.9):
        np.testing.assert_array_equal(quantiles[f"q{tau}"].to_numpy(), stats.poisson.ppf(tau, mean))
    assert model.dispersion_ is None


def test_tweedie_recovers_the_mean_and_the_dispersion():
    df, mu = _tweedie()
    model = _model("tweedie", power=1.5).fit(df)
    assert abs(model.predict(df)["Estimatedy"].mean() / df["y"].mean() - 1.0) < 0.05
    assert abs(model.dispersion_ - 2.0) / 2.0 < 0.15
    assert (df["y"] == 0).mean() > 0.6                         # a mostly-zero target


def test_tweedie_quantiles_are_non_negative_and_keep_the_mass_at_zero():
    train, _ = _tweedie(seed=0)
    test, _ = _tweedie(seed=1, n=4000)
    model = _model("tweedie", power=1.5).fit(train)
    taus = (0.05, 0.25, 0.5, 0.75, 0.95)
    quantiles = model.predict_quantiles(test, taus)
    values = quantiles.to_numpy()
    assert values.min() >= 0.0 and np.isfinite(values).all()
    assert (np.diff(values, axis=1) >= -1e-9).all()
    mean = model.predict(test)["Estimatedy"].to_numpy()
    p_zero = np.exp(-(mean ** 0.5) / (model.dispersion_ * 0.5))      # lambda = mu^(2-p) / (phi (2-p)), p = 1.5
    sure_zero = p_zero > 0.5
    assert sure_zero.sum() > 100
    assert (quantiles["q0.5"].to_numpy()[sure_zero] == 0.0).all()
    assert (quantiles["q0.5"].to_numpy()[p_zero < 0.45] > 0.0).all()


def test_tweedie_interval_covers():
    train, _ = _tweedie(seed=0)
    test, _ = _tweedie(seed=3, n=4000)
    model = _model("tweedie", power=1.5).fit(train)
    quantiles = model.predict_quantiles(test, (0.1, 0.9))
    y = test["y"].to_numpy()
    assert float(np.mean(y <= quantiles["q0.9"].to_numpy())) > 0.88
    assert float(np.mean(y <= quantiles["q0.9"].to_numpy())) < 0.94
    assert float(np.mean(y >= quantiles["q0.1"].to_numpy())) > 0.88


@pytest.mark.parametrize("loss", ["negative_binomial", "tweedie"])
def test_a_negative_target_is_rejected(loss):
    df, _ = _negative_binomial(n=500)
    df.loc[0, "y"] = -1
    with pytest.raises(ValueError, match="negative"):
        _model(loss).fit(df)


@pytest.mark.parametrize("power", [1.0, 2.0, 0.5, 3.0])
def test_the_tweedie_power_must_lie_in_one_two(power):
    with pytest.raises(ValueError, match="power"):
        build_strategy("tweedie", power=power)


def test_the_dispersion_must_be_positive():
    with pytest.raises(ValueError, match="dispersion"):
        build_strategy("negative_binomial", dispersion=0.0)


def test_other_losses_have_no_discrete_quantiles_and_no_dispersion():
    df, _ = _negative_binomial(n=500)
    model = ta.StaticTAM("y ~ l(x)", date_col="date").fit(df)
    assert model.dispersion_ is None
    with pytest.raises(RuntimeError, match="predict_quantiles"):
        model.predict_quantiles(df, (0.5,))


def test_the_mean_of_the_new_families_is_a_log_link_glm():
    df, mu = _negative_binomial(n=5000)
    fitted = _model("negative_binomial").fit(df).predict(df)["Estimatedy"].to_numpy()
    assert (fitted > 0).all()
    assert np.corrcoef(np.log(fitted), np.log(mu))[0, 1] > 0.99
