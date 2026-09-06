# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for the expectile <-> quantile bijection (Jones / Yao-Tong)."""
import numpy as np
import pytest

from tam.model.statistics.estimation import empirical_expectile, expectile_level_for_quantile


@pytest.mark.parametrize("sampler", ["normal", "lognormal"])
def test_expectile_at_h_alpha_matches_quantile(sampler):
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(20000) if sampler == "normal" else rng.lognormal(0.0, 0.6, 20000)
    for alpha in (0.1, 0.5, 0.9):
        level = expectile_level_for_quantile(sample, alpha)
        assert abs(empirical_expectile(sample, level) - np.quantile(sample, alpha)) < 0.05 * sample.std()


def test_median_maps_to_mean_expectile_for_symmetric():
    rng = np.random.default_rng(1)
    assert abs(expectile_level_for_quantile(rng.standard_normal(20000), 0.5) - 0.5) < 0.02


def test_invalid_levels_raise():
    sample = np.random.default_rng(2).standard_normal(1000)
    with pytest.raises(ValueError):
        empirical_expectile(sample, 1.5)
    with pytest.raises(ValueError):
        expectile_level_for_quantile(sample, 0.0)
