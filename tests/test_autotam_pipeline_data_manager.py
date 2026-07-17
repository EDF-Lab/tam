# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.autotam.pipeline.data_manager.DataManager``.

Covers the chronological split logic, the lag-injection path, and
the error raised when neither df_train nor explicit splits are provided.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.pipeline.data_manager import DataManager
from tam.model.autotam.pipeline.context import PipelineContext


def _frame(n=120):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "ds": dates,
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(15, 3, n),
    })


def test_prepare_splits_df_train_chronologically():
    mgr = DataManager("load ~ AutoPipe(temp)", train_fraction=0.70, dev_fraction=0.15)
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")

    assert isinstance(ctx, PipelineContext)
    assert ctx.target == "load"
    # The three splits must partition the dataset: fit 70%, dev 15%, val 15% (±1 row rounding).
    total = len(ctx.df_fit) + len(ctx.df_dev) + len(ctx.df_val)
    assert total == len(df)
    # Chronological order must be preserved.
    assert ctx.df_fit["ds"].max() <= ctx.df_dev["ds"].min()
    assert ctx.df_dev["ds"].max() <= ctx.df_val["ds"].min()


def test_prepare_accepts_explicit_splits():
    mgr = DataManager("load ~ AutoPipe(temp)")
    df = _frame(90)
    fit, dev, val = df.iloc[:60], df.iloc[60:75], df.iloc[75:]
    ctx = mgr.prepare(df_fit=fit, df_dev=dev, df_val=val, date_col="ds")
    assert ctx.target == "load"
    assert len(ctx.df_fit) == 60


def test_prepare_raises_without_train_or_explicit_splits():
    mgr = DataManager("load ~ AutoPipe(temp)")
    with pytest.raises(ValueError, match="df_train OR explicit"):
        mgr.prepare()


def test_prepare_injects_lag_columns():
    mgr = DataManager("load ~ AutoPipe(temp, load@7)")
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    # At least one lag column must appear after augmentation.
    lag_cols = [c for c in ctx.df_all_aug.columns if "_lag_" in c]
    assert len(lag_cols) >= 1


def test_prepare_populates_search_space():
    mgr = DataManager("load ~ AutoPipe(temp)")
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert len(ctx.search_space) > 0
    # The feature 'temp' must appear in the search space.
    assert any("temp" in k for k in ctx.search_space)
