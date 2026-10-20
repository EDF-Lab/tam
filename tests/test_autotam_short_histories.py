# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Short histories: what AutoTAM does when it is given very few rows, and what it says when it cannot work.

``df_train`` is split by fractions, so it works down to a handful of rows; explicit ``df_fit``, ``df_dev`` and ``df_val`` are used as given.
When a set is empty, or too small for any formula to be estimated, the error names the sets and their row counts and says what to change.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "temp": rng.normal(20, 5, n)})
    df["load"] = 100 + 2 * df["temp"] + rng.normal(0, 2, n)
    return df


def _model():
    return AutoTAM("load ~ AutoPipe(temp)", n_experts=1, pop_size=4, use_opera=False)


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _fit_explicit(fit_rows, dev_rows, val_rows):
    df = _frame(fit_rows + dev_rows + val_rows)
    model = _model()
    model.fit(df_fit=df.iloc[:fit_rows], df_dev=df.iloc[fit_rows:fit_rows + dev_rows], df_val=df.iloc[fit_rows + dev_rows:],
              date_col="ds", expansions=EXPANSIONS)
    return model


@pytest.mark.parametrize("sizes", [(0, 5, 5), (5, 0, 5), (5, 5, 0)])
def test_an_empty_split_is_named_with_the_row_counts(sizes):
    with pytest.raises(ValueError) as error:
        _fit_explicit(*sizes)
    message = str(error.value)
    assert "df_fit" in message and "df_dev" in message and "df_val" in message
    assert f"{sizes[0]}" in message and "df_train" in message


def test_too_few_rows_for_any_formula_fails_at_fit_with_the_row_counts():
    """It used to return an empty model after a printed warning; the error only came at predict()."""
    with pytest.raises(ValueError) as error:
        _fit_explicit(3, 2, 2)
    message = str(error.value)
    assert "No valid experts survived" in message
    assert "fit=3" in message and "dev=2" in message and "val=2" in message
    assert "df_train" in message


def test_df_train_with_too_few_rows_fails_at_fit_with_the_row_counts():
    df = _frame(8)
    with pytest.raises(ValueError) as error:
        _model().fit(df, date_col="ds", expansions=EXPANSIONS)
    message = str(error.value)
    assert "No valid experts survived" in message
    assert "fit=5" in message and "dev=1" in message and "val=2" in message


def test_the_smallest_explicit_splits_that_work_still_work():
    assert _fit_explicit(5, 2, 2).trained_experts


def test_df_train_with_twelve_rows_is_sized_from_the_data():
    df = _frame(12)
    model = _model()
    model.fit(df, date_col="ds", expansions=EXPANSIONS)
    assert model.trained_experts
    assert len(model.ctx.df_fit) + len(model.ctx.df_dev) + len(model.ctx.df_val) == 12
