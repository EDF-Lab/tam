# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
``TAMExtrapolationWarning``: a forecast that leaves the trained range of a non-linear effect (or holds a categorical level
training never saw) says so, once per model and feature. The defaults do not change (D24); ``l()`` never warns.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.common.exceptions import collect_extrapolation


def _data(n=300, seed=0, n_groups=1):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(n_groups):
        df = pd.DataFrame({"date": pd.date_range("2022-01-03", periods=n, freq="D"), "g": g})
        df["dow"] = df["date"].dt.dayofweek
        df["x"] = rng.normal(size=n)
        df["z"] = rng.normal(size=n)
        df["y"] = 2 * df["x"] + np.sin(df["z"]) + 0.3 * df["dow"] + 0.05 * rng.standard_normal(n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if issubclass(w.category, ta.TAMExtrapolationWarning)]


def test_a_spline_beyond_the_trained_range_warns_once_and_names_the_feature():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + s(z, k=6)", date_col="date").fit(df)
    shifted = df.assign(z=df["z"] + 5.0)
    first = _caught(lambda: model.predict(shifted))
    second = _caught(lambda: model.predict(shifted))
    assert len(first) == 1 and "'z'" in str(first[0].message) and "extrapolate='constant'" in str(first[0].message)
    assert second == []


def test_a_linear_effect_never_warns():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + c(dow)", date_col="date").fit(df)
    assert _caught(lambda: model.predict(df.assign(x=df["x"] * 10.0))) == []


def test_inside_the_range_does_not_warn():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + s(z, k=6)", date_col="date").fit(df)
    assert _caught(lambda: model.predict(df)) == []


def test_a_categorical_level_unseen_in_training_warns():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + c(dow, n_cat=7)", date_col="date").fit(df[df["dow"] < 5])
    caught = _caught(lambda: model.predict(df))
    assert len(caught) == 1 and "[5, 6]" in str(caught[0].message)


def test_every_level_seen_does_not_warn():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + c(dow)", date_col="date").fit(df)
    assert _caught(lambda: model.predict(df)) == []


def test_a_tensor_product_margin_and_each_group_own_range():
    df = _data(n_groups=2)
    model = ta.StaticTAM(formula="y ~ te(s(x, k=4), s(z, k=4))", group_col="g", date_col="date").fit(df)
    future = df.copy()
    future.loc[future["g"] == 1, "x"] += 6.0                  # only group 1 leaves its own range
    caught = _caught(lambda: model.predict(future))
    assert len(caught) == 1 and "'x'" in str(caught[0].message)


def test_the_warning_can_be_escalated_in_ci():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + s(z, k=6)", date_col="date").fit(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ta.TAMExtrapolationWarning)
        with pytest.raises(ta.TAMExtrapolationWarning):
            model.predict(df.assign(z=df["z"] + 5.0))


def test_a_collected_block_records_the_feature_and_raises_nothing():
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + s(z, k=6)", date_col="date").fit(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ta.TAMExtrapolationWarning)
        with collect_extrapolation() as features:
            model.predict(df.assign(z=df["z"] + 5.0))
    assert list(features) == ["z"]


def test_adaptive_aggregates_its_windows_into_one_warning():
    df = _data(n=160)
    df.loc[df.index >= 100, "z"] += 8.0                       # leaves the range of the training windows
    model = ta.AdaptiveTAM(adaptive_formula="y ~ s(z, k=6)", update_interval_periods=10, training_window_periods=40,
                           steps_per_period=1, date_col="date")
    caught = _caught(lambda: model.predict_online(df))
    assert len(caught) == 1 and "windows" in str(caught[0].message) and "'z'" in str(caught[0].message)
    assert _caught(lambda: model.predict_online(df)) == []


def test_adaptive_with_a_linear_effect_does_not_warn():
    df = _data(n=160)
    df.loc[df.index >= 100, "x"] += 8.0
    model = ta.AdaptiveTAM(adaptive_formula="y ~ l(x)", update_interval_periods=10, training_window_periods=40,
                           steps_per_period=1, date_col="date")
    assert _caught(lambda: model.predict_online(df)) == []


def test_a_tiny_overshoot_of_the_trained_range_does_not_warn():
    """A new sample can land a hair beyond the training minimum or maximum: below 1% of the half-range it is not reported."""
    df = _data()
    model = ta.StaticTAM(formula="y ~ l(x) + s(z, k=6)", date_col="date").fit(df)
    half_range = (df["z"].max() - df["z"].min()) / 2
    tiny = df.assign(z=df["z"].where(df["z"] != df["z"].max(), df["z"].max() + 0.005 * half_range))
    assert _caught(lambda: model.predict(tiny)) == []
    big = df.assign(z=df["z"].where(df["z"] != df["z"].max(), df["z"].max() + 0.05 * half_range))
    assert len(_caught(lambda: model.predict(big))) == 1


# ------------------------------------------------------------------ AdaptiveTAM: a categorical level absent from the training rows of its window
def _holiday_data(n=160, first_seen=120, level=3):
    """A weekday-like cycle 0..2 plus one level (`level`) that appears for the first time on day `first_seen` and then every 20 days."""
    df = _data(n=n)
    df["h"] = (np.arange(n) % 3).astype(float)
    df.loc[(df.index >= first_seen) & ((df.index - first_seen) % 20 == 0), "h"] = float(level)
    return df


def _adaptive(formula, **kwargs):
    return ta.AdaptiveTAM(adaptive_formula=formula, update_interval_periods=10, training_window_periods=40, steps_per_period=1,
                          date_col="date", **kwargs)


@pytest.mark.parametrize("formula", ["y ~ l(x) + c(h, n_cat=4, topo='nominal')", "y ~ l(x) + c(h, topo='nominal')"])
def test_adaptive_warns_once_for_a_level_absent_from_the_training_rows_of_its_window(formula):
    df = _holiday_data()
    model = _adaptive(formula)
    caught = _caught(lambda: model.predict_online(df))
    levels = [w for w in caught if "'h'" in str(w.message) and "training rows" in str(w.message)]
    assert len(levels) == 1, [str(w.message) for w in caught]
    message = str(levels[0].message)
    assert "[3.0]" in message and "windows" in message
    assert _caught(lambda: model.predict_online(df)) == []                      # once per model


def test_adaptive_is_silent_when_every_level_was_seen_in_its_window():
    df = _holiday_data(first_seen=0)                                            # the level occurs from the first rows on, every 20 days
    model = _adaptive("y ~ l(x) + c(h, n_cat=4, topo='nominal')")
    assert _caught(lambda: model.predict_online(df)) == []


def test_adaptive_without_a_categorical_term_is_silent_about_levels():
    df = _holiday_data()
    model = _adaptive("y ~ l(x) + l(h)")
    assert _caught(lambda: model.predict_online(df)) == []


def test_adaptive_level_warning_is_collected_not_raised_inside_collect_extrapolation():
    df = _holiday_data()
    model = _adaptive("y ~ l(x) + c(h, n_cat=4, topo='nominal')")
    with collect_extrapolation() as collected:
        assert _caught(lambda: model.predict_online(df)) == []
    assert any("h" in key for key in collected)
