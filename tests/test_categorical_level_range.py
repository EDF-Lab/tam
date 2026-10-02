# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A categorical code is a level index: it is normalised on the full level range ``0 .. n_cat - 1``, not on the levels present in
the training rows. A level absent from training keeps its own column (penalised towards 0) instead of landing on an edge level.
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.spectrum import categorical_ranges, create_effects_from_parsed_terms

LEVEL_EFFECT = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0])


def _week(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2022-01-03", periods=n, freq="D")})
    df["dow"] = df["date"].dt.dayofweek
    df["x"] = rng.normal(size=n)
    df["y"] = 2.0 * df["x"] + LEVEL_EFFECT[df["dow"]]
    return df


FORMULA = "y ~ l(x) + c(dow, n_cat=7, topo='nominal', ap=-3)"


def test_unseen_level_keeps_its_own_column():
    df = _week()
    model = ta.StaticTAM(formula=FORMULA, date_col="date").fit(df[df["dow"] < 5])
    saturday = df[df["dow"] == 5].head(5)
    friday = saturday.assign(dow=4)
    pred_sat = model.predict(saturday)["Estimatedy"].to_numpy()
    pred_fri = model.predict(friday)["Estimatedy"].to_numpy()
    # Saturday's coefficient is ~0 (never trained), Friday's is the trained one: the two differ by Friday's effect
    gap = pred_fri - pred_sat
    assert np.all(gap > 1.5), gap


def test_training_levels_are_unchanged_by_the_unseen_ones():
    """Rows of the trained levels are predicted as before: the level range only moves where an unseen level goes."""
    df = _week()
    train = df[df["dow"] < 5]
    full = ta.StaticTAM(formula=FORMULA, date_col="date").fit(train)
    centred = full.predict(train)["Estimatedy"].to_numpy()
    assert np.sqrt(np.mean((centred - train["y"].to_numpy()) ** 2)) < 0.05


def test_every_level_in_training_changes_nothing():
    df = _week()
    model = ta.StaticTAM(formula=FORMULA, date_col="date").fit(df)
    params = next(iter(model.norm_params_.values()))
    assert (params["min"]["dow"], params["max"]["dow"]) == (0.0, 6.0)
    assert np.sqrt(np.mean((model.predict(df)["Estimatedy"].to_numpy() - df["y"].to_numpy()) ** 2)) < 0.05


def test_levels_missing_at_the_bottom_still_use_the_full_range():
    df = _week()
    model = ta.StaticTAM(formula=FORMULA, date_col="date").fit(df[df["dow"] >= 2])
    params = next(iter(model.norm_params_.values()))
    assert (params["min"]["dow"], params["max"]["dow"]) == (0.0, 6.0)


def test_codes_from_one_to_n_cat_keep_their_own_range():
    """Months coded 1..12 with n_cat=12 do not fit in 0..11: they are normalised on (1, 12), as before."""
    df = _week(n=400)
    df["month"] = df["date"].dt.month
    model = ta.StaticTAM(formula="y ~ l(x) + c(month, n_cat=12, topo='fourier')", date_col="date").fit(df)
    params = next(iter(model.norm_params_.values()))
    assert (params["min"]["month"], params["max"]["month"]) == (1.0, 12.0)


def test_fractional_codes_keep_the_min_max_rule():
    """A month coded as (month - 1) / 12 and read by a Fourier topology is a coordinate, not a level index."""
    df = _week(n=400)
    df["season"] = (df["date"].dt.month - 1) / 12.0
    model = ta.StaticTAM(formula="y ~ l(x) + c(season, n_cat=12, topo='fourier')", date_col="date").fit(df)
    params = next(iter(model.norm_params_.values()))
    assert (params["min"]["season"], params["max"]["season"]) == (0.0, 11 / 12)


def test_an_inferred_n_cat_keeps_the_min_max_rule():
    """Without n_cat in the formula the level set is not known (n_cat = training maximum + 1): codes 1..12 keep (1, 12)."""
    df = _week(n=400)
    df["month"] = df["date"].dt.month
    model = ta.StaticTAM(formula="y ~ l(x) + c(month, topo='fourier')", date_col="date").fit(df)
    params = next(iter(model.norm_params_.values()))
    assert (params["min"]["month"], params["max"]["month"]) == (1.0, 12.0)


def test_a_feature_shared_with_another_effect_keeps_the_min_max_rule():
    effects = create_effects_from_parsed_terms(
        ta.StaticTAM(formula="y ~ s(dow, k=5) + c(dow, n_cat=7)", date_col="date").parsed_terms_,
        token_values={}, default_alpha_p=-3.0, data_info={"dow": 7})
    assert categorical_ranges(effects) == {}


def test_a_numeric_feature_is_never_fixed():
    df = _week()
    model = ta.StaticTAM(formula=FORMULA, date_col="date").fit(df[df["dow"] < 5])
    params = next(iter(model.norm_params_.values()))
    assert params["min"]["x"] == pytest.approx(df.loc[df["dow"] < 5, "x"].min())
    assert params["max"]["x"] == pytest.approx(df.loc[df["dow"] < 5, "x"].max())


def _info(fn):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if "TAM [Info]" in str(w.message)]


def test_training_says_which_levels_are_missing():
    df = _week()
    messages = _info(lambda: ta.StaticTAM(formula=FORMULA, date_col="date").fit(df[df["dow"] < 5]))
    assert len(messages) == 1 and "'dow'" in messages[0] and "[5, 6]" in messages[0] and "regularized to 0" in messages[0]


def test_training_with_every_level_or_an_inferred_n_cat_does_not_say_anything():
    df = _week()
    assert _info(lambda: ta.StaticTAM(formula=FORMULA, date_col="date").fit(df)) == []
    assert _info(lambda: ta.StaticTAM(formula="y ~ l(x) + c(dow)", date_col="date").fit(df[df["dow"] < 5])) == []
