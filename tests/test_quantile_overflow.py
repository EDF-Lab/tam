# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A quantile that overflows float64 on the response scale (log target) is ``inf`` with an explicit ``UserWarning``,
never a silent numpy ``RuntimeWarning``.
"""

import warnings

import numpy as np
import pytest

from test_distributional import _fit, _heteroscedastic_lognormal
from tam.model.statistics.estimation import _distributional as dist


@pytest.fixture
def model_and_data():
    rng = np.random.default_rng(11)
    model = _fit(rng, dist_kwargs={"tail_family": "normal"})
    test, _, _ = _heteroscedastic_lognormal(rng, 200)
    return model, test


def _with_huge_sigma(monkeypatch, row, sigma=1e6):
    original = dist.mu_sigma

    def patched(model, data):
        mu_hat, sigma_hat = original(model, data)
        sigma_hat = sigma_hat.copy()
        sigma_hat[row] = sigma
        return mu_hat, sigma_hat

    monkeypatch.setattr(dist, "mu_sigma", patched)


def test_overflowing_quantile_is_inf_with_an_explicit_warning(model_and_data, monkeypatch):
    model, test = model_and_data
    _with_huge_sigma(monkeypatch, row=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # a numpy overflow warning would fail the test
        with pytest.warns(UserWarning, match=r"overflow.*0\.9.*1 row"):
            q = model.predict_quantile(test, 0.9)
    assert np.isinf(q[3]) and q[3] > 0
    assert np.isfinite(np.delete(q, 3)).all()


def test_the_lower_tail_underflows_to_zero_without_warning(model_and_data, monkeypatch):
    model, test = model_and_data
    _with_huge_sigma(monkeypatch, row=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = model.predict_quantile(test, 0.1)
    assert q[3] == 0.0


def test_median_with_an_overflowing_location_warns_too(model_and_data, monkeypatch):
    model, test = model_and_data
    original = dist.mu_sigma
    monkeypatch.setattr(dist, "mu_sigma", lambda m, d: (np.where(np.arange(len(d)) == 2, 800.0, original(m, d)[0]), original(m, d)[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(UserWarning, match=r"overflow.*median.*1 row"):
            m = model.predict_median(test)
    assert np.isinf(m[2])


def test_finite_quantiles_do_not_warn(model_and_data):
    model, test = model_and_data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = model.predict_quantile(test, [0.1, 0.5, 0.9])
    assert np.isfinite(q).all()
