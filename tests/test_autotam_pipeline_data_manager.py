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


def test_prepare_with_group_col_preserves_chronological_order():
    mgr = DataManager("load ~ AutoPipe(temp)")
    dates = pd.date_range("2023-01-01", periods=20, freq="15min")
    tods = [d.hour * 4 + d.minute // 15 for d in dates]
    df = pd.DataFrame({
        "ds": dates,
        "load": np.arange(20, dtype=float),
        "temp": np.linspace(10, 20, 20),
        "tod": tods,
    })
    ctx = mgr.prepare(df_train=df, date_col="ds", group_col="tod")
    assert ctx.df_fit["ds"].is_monotonic_increasing
    assert ctx.df_val["ds"].is_monotonic_increasing


def test_prepare_accepts_explicit_splits():
    mgr = DataManager("load ~ AutoPipe(temp)")
    df = _frame(90)
    fit, dev, val = df.iloc[:60], df.iloc[60:75], df.iloc[75:]
    ctx = mgr.prepare(df_fit=fit, df_dev=dev, df_val=val, date_col="ds")
    assert ctx.target == "load"
    assert len(ctx.df_fit) == 60


def test_prepare_explicit_splits_with_overlapping_index_made_disjoint():
    mgr = DataManager("load ~ AutoPipe(temp)")
    df = _frame(90)
    fit = df.iloc[:60].reset_index(drop=True)
    dev = df.iloc[60:75].reset_index(drop=True)
    val = df.iloc[75:].reset_index(drop=True)
    ctx = mgr.prepare(df_fit=fit, df_dev=dev, df_val=val, date_col="ds")

    # Indices must be strictly disjoint between fit, dev, and val
    assert len(ctx.df_fit.index.intersection(ctx.df_dev.index)) == 0
    assert len(ctx.df_dev.index.intersection(ctx.df_val.index)) == 0
    assert len(ctx.df_fit.index.intersection(ctx.df_val.index)) == 0

    # Combined df_cont_val must have a unique index
    df_cont = pd.concat([ctx.df_fit, ctx.df_dev, ctx.df_val])
    assert df_cont.index.is_unique


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


def test_prepare_stores_normalized_mandatory_terms():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_terms=["s(temp, k=10)"])
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert ctx.mandatory_terms == ["s(temp, k=10)"]


def test_prepare_accepts_single_string_mandatory_term():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_terms="s(temp, k=10)")
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert ctx.mandatory_terms == ["s(temp, k=10)"]


def test_prepare_raises_on_syntactically_invalid_mandatory_term():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_terms=["invalid_syntax_term((("])
    df = _frame(120)
    with pytest.raises(ValueError, match="Invalid mandatory term"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_raises_on_semantic_unknown_feature_in_mandatory_term():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_terms=["s(unknown_feature, k=10)"])
    df = _frame(120)
    with pytest.raises(ValueError, match="unknown_feature"):
        mgr.prepare(df_train=df, date_col="ds")

def test_prepare_raises_on_semantic_unknown_feature_in_mandatory_variable():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_variables=["unknown_feature"])
    df = _frame(120)
    with pytest.raises(ValueError, match="unknown_feature"):
        mgr.prepare(df_train=df, date_col="ds")

def test_prepare_accepts_single_string_mandatory_variable():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_variables="temp")
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert ctx.mandatory_variables == ["temp"]

def test_prepare_stores_normalized_mandatory_variables():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_variables=["temp"])
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert ctx.mandatory_variables == ["temp"]


def test_prepare_rejects_duplicate_mandatory_terms_exact():
    mgr = DataManager("load ~ AutoPipe(temp)", mandatory_terms=["s(temp, k=10)", "s(temp, k=10)"])
    df = _frame(120)
    with pytest.raises(ValueError, match="Duplicate mandatory term"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_rejects_duplicate_mandatory_terms_under_canonical_equivalence():
    mgr = DataManager(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["c(temp, topo='nominal', n_cat=7)", "c(temp, n_cat=7, topo='nominal')"],
    )
    df = _frame(120)
    with pytest.raises(ValueError, match="Duplicate mandatory term"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_rejects_exceeding_max_active_effects_per_feature():
    mgr = DataManager(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["s(temp, k=5)", "l(temp)", "f(temp, m=3)"],
    )
    df = _frame(120)
    with pytest.raises(ValueError, match="MAX_ACTIVE_EFFECTS_PER_FEATURE"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_allows_up_to_max_active_effects_per_feature():
    mgr = DataManager(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["s(temp, k=5)", "l(temp)"],
    )
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert len(ctx.mandatory_terms) == 2


def test_prepare_rejects_exceeding_max_tensor_terms():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=120, freq="D"),
        "load": rng.normal(100, 10, 120),
        "temp": rng.normal(15, 3, 120),
        "hum": rng.normal(50, 10, 120),
        "pressure": rng.normal(1013, 10, 120),
    })
    mgr = DataManager(
        "load ~ AutoPipe(temp, hum, pressure)",
        mandatory_terms=[
            "te(s(temp), s(hum))",
            "te(s(temp), s(pressure))",
            "te(s(hum), s(pressure))",
        ],
    )
    with pytest.raises(ValueError, match="MAX_TENSOR_TERMS"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_allows_up_to_max_tensor_terms():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=120, freq="D"),
        "load": rng.normal(100, 10, 120),
        "temp": rng.normal(15, 3, 120),
        "hum": rng.normal(50, 10, 120),
    })
    mgr = DataManager(
        "load ~ AutoPipe(temp, hum)",
        mandatory_terms=[
            "te(s(temp), s(hum))",
            "te(l(temp), l(hum))",
        ],
    )
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert len(ctx.mandatory_terms) == 2


def test_prepare_allows_tensor_terms_with_parenthesized_kwargs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=120, freq="D"),
        "load": rng.normal(100, 10, 120),
        "temp": rng.normal(15, 3, 120),
        "hum": rng.normal(50, 10, 120),
    })
    mgr = DataManager(
        "load ~ AutoPipe(temp, hum)",
        mandatory_terms=["te(s(temp), s(hum), bs=('cr', 'ps'))"],
    )
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert len(ctx.mandatory_terms) == 1


def test_prepare_rejects_semantic_unknown_feature_in_tensor_subterm():
    df = _frame(120)
    mgr = DataManager(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["te(s(temp), s(unknown_feat))"],
    )
    with pytest.raises(ValueError, match="unknown_feat"):
        mgr.prepare(df_train=df, date_col="ds")


def test_prepare_populates_canonical_to_verbatim_mandatory():
    mgr = DataManager(
        "load ~ AutoPipe(temp)",
        mandatory_terms=["c(temp, topo='nominal', n_cat=7)"],
    )
    df = _frame(120)
    ctx = mgr.prepare(df_train=df, date_col="ds")
    assert ctx.mandatory_terms == ["c(temp, topo='nominal', n_cat=7)"]
    assert "c(temp, n_cat=7, topo='nominal')" in ctx.canonical_to_verbatim_mandatory
    assert (
        ctx.canonical_to_verbatim_mandatory["c(temp, n_cat=7, topo='nominal')"]
        == "c(temp, topo='nominal', n_cat=7)"
    )
