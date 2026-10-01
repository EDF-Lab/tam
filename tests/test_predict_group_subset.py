# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A grouped model predicts and decomposes a frame holding only some of its groups.

The rows of a group must not depend on which other groups are in the frame: same values, same order, same index.
A group never seen in training raises a ``ValueError`` naming it.
"""

import numpy as np
import pandas as pd
import pytest

import tam as ta

FORMULAS = ["y ~ l(x) + s(x)", "y ~ s(x) + c(k)"]
N_PER_GROUP = 60


def _data(seed=0, groups=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    frames = []
    for j, g in enumerate(groups):
        x = rng.uniform(-2, 2, N_PER_GROUP)
        frames.append(pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=N_PER_GROUP, freq="D"),
            "g": g, "x": x, "k": rng.integers(0, 3, N_PER_GROUP),
            "y": np.sin(x) + j + rng.normal(0, 0.1, N_PER_GROUP)}))
    return pd.concat(frames, ignore_index=True)


def _fit(formula, df):
    return ta.StaticTAM(formula=formula, group_col="g", date_col="date").fit(df)


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("group", ["a", "b", "c"])
def test_one_group_frame_equals_the_same_rows_in_the_full_frame(formula, group):
    df = _data()
    model = _fit(formula, df)
    full = model.predict(df)
    rows = df[df["g"] == group]
    sub = model.predict(rows)
    assert list(sub.index) == list(rows.index)
    np.testing.assert_allclose(sub["Estimatedy"].to_numpy(), full.loc[rows.index, "Estimatedy"].to_numpy(), rtol=1e-12, atol=1e-12)
    assert not sub["Estimatedy"].isna().any()


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("group", ["a", "b", "c"])
def test_decomposition_of_a_one_group_frame_equals_the_full_frame(formula, group):
    df = _data()
    model = _fit(formula, df)
    full = model.decompose_prediction(df)
    rows = df[df["g"] == group]
    sub = model.decompose_prediction(rows)
    effects = [c for c in full.columns if c.startswith("effect_")]
    assert effects
    assert list(sub.index) == list(rows.index)
    assert not sub[effects].isna().any().any()
    np.testing.assert_allclose(sub[effects].to_numpy(), full.loc[rows.index, effects].to_numpy(), rtol=1e-12, atol=1e-12)


def test_two_of_three_groups_keep_input_order_and_index():
    df = _data()
    model = _fit("y ~ l(x) + s(x)", df)
    full = model.predict(df)
    rows = df[df["g"].isin(["a", "c"])].sample(frac=1.0, random_state=1)  # shuffled, non-contiguous index
    rows.index = rows.index + 1000
    sub = model.predict(rows)
    assert list(sub.index) == list(rows.index)
    assert not sub["Estimatedy"].isna().any()
    expected = full.loc[rows.index - 1000, "Estimatedy"].to_numpy()
    np.testing.assert_allclose(sub["Estimatedy"].to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_a_group_never_seen_in_training_raises_and_is_named():
    df = _data(groups=("a", "b"))
    model = _fit("y ~ l(x) + s(x)", df)
    new = _data(groups=("a", "zzz"))
    with pytest.raises(ValueError, match="zzz"):
        model.predict(new)
    with pytest.raises(ValueError, match="zzz"):
        model.decompose_prediction(new)


def test_the_full_frame_is_unchanged_by_the_fix():
    df = _data()
    model = _fit("y ~ l(x) + s(x)", df)
    out = model.predict(df)
    assert len(out) == len(df) and not out["Estimatedy"].isna().any()
