# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Distribution families: the profile of the target decides which GLM losses are offered, and ``family="gaussian"`` is the search as it was.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.families import candidate_losses, check_family_mode
from tam.model.autotam.target_profile import profile_target

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(kind, n=220, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    if kind == "lognormal":
        y = np.exp(2.0 + 0.5 * x1 + rng.normal(scale=0.3, size=n))
    elif kind == "zeros":
        y = np.where(rng.random(n) < 0.3, 0.0, np.exp(1.5 + 0.4 * x1 + rng.normal(scale=0.3, size=n)))
    elif kind == "counts":
        y = rng.poisson(np.exp(1.0 + 0.4 * x1)).astype(float)
    elif kind == "negatives":
        y = 3 * x1 + rng.normal(scale=0.5, size=n)
    else:
        y = 50 + 4 * x1 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x1, "y": y})


@pytest.mark.parametrize("kind, expected", [("lognormal", ["gamma", "huber"]), ("zeros", ["tweedie"]), ("counts", ["poisson", "negative_binomial"])])
def test_the_profile_proposes_the_family_the_data_allow(kind, expected):
    assert candidate_losses(profile_target(_frame(kind)["y"])) == expected


def test_no_log_family_is_proposed_for_negative_values_or_a_gaussian_target():
    assert candidate_losses(profile_target(_frame("negatives")["y"])) == []
    assert candidate_losses(profile_target(_frame("gaussian")["y"])) == []


def test_a_short_series_a_sentinel_or_nothing_proposes_nothing():
    assert candidate_losses({}) == []
    assert candidate_losses(profile_target(_frame("lognormal", n=10)["y"])) == []


def test_the_family_mode_is_checked():
    assert check_family_mode("auto") == "auto"
    with pytest.raises(ValueError, match="family must be one of"):
        check_family_mode("poisson")


def _fit(tmp_path, kind, **kwargs):
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=3, pop_size=6, use_opera=False, seed=4, export_dir=str(tmp_path), **kwargs)
    return model.fit(_frame(kind), date_col="ds", expansions=EXPANSIONS)


def test_a_positive_skewed_target_gets_a_gamma_expert_scored_in_the_response_scale(tmp_path):
    model = _fit(tmp_path, "lognormal", family="auto")
    tried = [row for row in model.chronological_test_log if "Family_gamma" in str(row.get("Model_Name"))]
    assert tried and all(np.isfinite(row["Validation_RMSE"]) for row in tried)
    forecast = model.predict(_frame("lognormal").iloc[:5], date_col="ds")
    assert (forecast.drop(columns=[c for c in forecast.columns if c == "ds"]).select_dtypes("number").to_numpy() > 0).all()


def test_data_with_zeros_and_negatives_get_no_log_family_and_no_error(tmp_path):
    model = _fit(tmp_path, "negatives", family="auto")
    assert not any("Family_" in str(row.get("Model_Name")) for row in model.chronological_test_log) and model.trained_experts


def test_gaussian_family_is_the_search_as_it_was(tmp_path):
    one = _fit(tmp_path, "lognormal")
    two = _fit(tmp_path, "lognormal", family="gaussian")
    assert one.architectures_ == two.architectures_ and not any("Family_" in str(r.get("Model_Name")) for r in two.chronological_test_log)
