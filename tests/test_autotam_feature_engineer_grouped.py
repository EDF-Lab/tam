# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Equivalence tests for the grouped rolling-mean and EWMA features of
``tam.model.autotam.feature_engineer.FeatureEngineer``: the native grouped window operations must
reproduce the former per-group ``transform(lambda ...)`` exactly, including missing values,
duplicate index labels and missing group keys.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.feature_engineer import FeatureEngineer

WINDOW, ALPHA = 7, 0.25


def _lambda_reference(df, group_col, col):
    roll = df.groupby(group_col)[col].transform(lambda x: x.rolling(window=WINDOW, min_periods=1).mean())
    ewm = df.groupby(group_col)[col].transform(lambda x: x.ewm(alpha=ALPHA, adjust=False, ignore_na=True).mean())
    return roll.to_numpy(), ewm.to_numpy()


def _engineered(df, group_col, col):
    engineer = FeatureEngineer()
    engineer.learned_temporal_params[col] = {"window_size": WINDOW, "dynamic_alpha": ALPHA}
    engineer.is_fitted = True
    out = engineer._create_temporal_features(df.copy(), [col], group_col)
    return out[f"{col}_rolling_mean_{WINDOW}_steps"].to_numpy(), out[f"{col}_ewma_alpha{int(ALPHA * 100)}"].to_numpy()


def _panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": rng.integers(0, 5, n).astype(float), "x": rng.normal(size=n)})
    df.loc[rng.choice(n, 40, replace=False), "x"] = np.nan
    return df


@pytest.mark.parametrize("variant", ["plain", "duplicate_index", "missing_group_keys"])
def test_grouped_features_match_the_per_group_lambda(variant):
    df = _panel()
    if variant == "duplicate_index":
        df.index = np.repeat(np.arange(len(df) // 2), 2)
    if variant == "missing_group_keys":
        df.loc[df.index[:15], "g"] = np.nan

    expected_roll, expected_ewm = _lambda_reference(df, "g", "x")
    roll, ewm = _engineered(df, "g", "x")

    np.testing.assert_allclose(roll, expected_roll, equal_nan=True)
    np.testing.assert_allclose(ewm, expected_ewm, equal_nan=True)
