# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
What the data profiler says about the target: missing-value sentinels, bounds, cumulative series and a family hint.

Some facts about a target make any fit wrong when nobody looks: a station's missing-value code (-9999) fitted as a value, a counter that only grows,
a count or a proportion modelled as a Gaussian. ``profile_target`` reads them; ``AutoTAM`` reports them (``target_profile_``), warns about a sentinel,
and replaces it by a missing value only when asked (``replace_sentinels=True``).
"""

import inspect
import logging
import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import target_profile as tp
from tam.model.autotam.auto_tam import AutoTAM


def test_the_constructor_has_a_defaulted_replace_sentinels_argument():
    assert inspect.signature(AutoTAM.__init__).parameters["replace_sentinels"].default is False


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #

def _rng():
    return np.random.default_rng(7)


def test_a_clean_gaussian_series_has_no_hint_beyond_gaussian():
    y = pd.Series(_rng().normal(100, 10, 500))
    profile = tp.profile_target(y)
    assert profile["hints"] == ["gaussian"] and profile["sentinels"] == []


def test_counts_are_recognised():
    profile = tp.profile_target(pd.Series(_rng().poisson(4.0, 500)))
    assert profile["integer_share"] == 1.0 and profile["hints"] == ["count"]


def test_zero_mass_with_positive_values_is_tweedie():
    y = np.where(_rng().random(500) < 0.4, 0.0, _rng().gamma(2.0, 3.0, 500))
    assert tp.profile_target(pd.Series(y))["hints"] == ["tweedie"]


def test_a_proportion_is_bounded():
    y = _rng().beta(2.0, 5.0, 500)
    profile = tp.profile_target(pd.Series(y))
    assert profile["hints"] == ["bounded"] and profile["bounds"] == (0.0, 1.0)


def test_a_percentage_is_bounded_at_100():
    y = np.clip(_rng().normal(60, 25, 500), 0, 100)
    assert tp.profile_target(pd.Series(y))["bounds"] == (0.0, 100.0)


def test_a_cumulative_series_is_monotone():
    y = np.cumsum(_rng().poisson(3.0, 400)).astype(float)
    profile = tp.profile_target(pd.Series(y))
    assert profile["monotone_share"] >= 0.99 and profile["hints"] == ["cumulative"]


def test_a_skewed_positive_series_suggests_a_log():
    assert tp.profile_target(pd.Series(_rng().lognormal(1.0, 0.9, 500)))["hints"] == ["log"]


def test_a_skewed_series_with_negatives_suggests_asinh():
    y = -1.0 + 4.0 * np.sinh(1.2 * _rng().standard_normal(500))
    assert tp.profile_target(pd.Series(y))["hints"] == ["asinh"]


def test_a_missing_value_code_is_a_sentinel():
    y = _rng().normal(20, 5, 500)
    y[::100] = -9999.0
    profile = tp.profile_target(pd.Series(y))
    assert profile["sentinels"] == [-9999.0]


def test_a_single_extreme_value_is_not_a_sentinel():
    y = _rng().normal(20, 5, 500)
    y[10] = 500.0
    assert tp.profile_target(pd.Series(y))["sentinels"] == []


def test_masking_a_sentinel_gives_a_missing_value_and_leaves_the_input_alone():
    df = pd.DataFrame({"y": [1.0, -9999.0, 3.0, -9999.0]})
    out = tp.mask_sentinels(df, "y", [-9999.0])
    assert out["y"].isna().tolist() == [False, True, False, True]
    assert df["y"].tolist() == [1.0, -9999.0, 3.0, -9999.0]


# --------------------------------------------------------------------------- #
# In the fit
# --------------------------------------------------------------------------- #

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _series(n=160, sentinels=True):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    df["y"] = 20.0 + 3.0 * df["x1"] + rng.normal(scale=0.5, size=n)
    if sentinels:
        df.loc[df.index[::20], "y"] = -9999.0
    return df


def _fit(df, **kwargs):
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=4, use_opera=False, **kwargs)
    model.fit(df, date_col="ds", expansions=EXPANSIONS)
    return model


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_the_profile_is_reported_after_the_fit():
    model = _fit(_series(sentinels=False))
    assert model.target_profile_["hints"] == ["gaussian"]


def test_a_sentinel_is_reported_with_a_warning_and_left_in_the_data_by_default(caplog):
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam.pipeline.data_manager"):
        model = _fit(_series())
    assert model.target_profile_["sentinels"] == [-9999.0]
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("-9999" in m and "replace_sentinels" in m for m in messages)
    assert (model.ctx.df_all_aug["y"] == -9999.0).any()


def test_replace_sentinels_turns_them_into_missing_values_before_the_fit():
    model = _fit(_series(), replace_sentinels=True)
    assert not (model.ctx.df_all_aug["y"] == -9999.0).any()
    assert model.trained_experts


def test_a_clean_series_warns_about_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam.pipeline.data_manager"):
        _fit(_series(sentinels=False))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "sentinel" in r.getMessage().lower()]
