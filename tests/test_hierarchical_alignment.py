# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``HierarchicalTAM`` matches its nodes on ``(group, date)``, not on row position.

``align="strict"`` (default) raises on a mismatch and lists the first missing keys per node, ``align="inner"`` keeps the common keys and
warns with the rows dropped per node. A parent whose target is not the sum of its children's is reported once at fit.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tam.model.hierarchical import HierarchicalTAM

FORMULA = "load ~ l(temperature) + s(temperature, k=5)"
STRUCTURE = {"Total": ["RegionA", "RegionB"]}


def _data(n=60, seed=0, additive=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n)
    temp = rng.uniform(-5, 30, n)
    frames = []
    loads = {"RegionA": 60.0 + temp + rng.normal(0, 2, n), "RegionB": 40.0 + 0.5 * temp + rng.normal(0, 2, n)}
    loads["Total"] = loads["RegionA"] + loads["RegionB"] if additive else 100.0 + 1.5 * temp + rng.normal(0, 4, n)
    for node in ("Total", "RegionA", "RegionB"):
        frames.append(pd.DataFrame({"timestamp": dates, "node": node, "temperature": temp, "load": loads[node]}))
    return pd.concat(frames, ignore_index=True)


def _model(**kwargs):
    return HierarchicalTAM(structure=STRUCTURE, formulas=FORMULA, node_col="node", date_col="timestamp", **kwargs)


def _drop_days(df, node, days):
    return df[~((df["node"] == node) & df["timestamp"].isin(days))].reset_index(drop=True)


def test_the_default_is_strict():
    assert _model().align == "strict"


def test_an_unknown_alignment_mode_is_refused():
    with pytest.raises(ValueError, match="align"):
        _model(align="outer")


def test_strict_raises_and_lists_the_first_missing_dates_per_node():
    df = _data()
    days = pd.to_datetime(["2022-01-05", "2022-01-09", "2022-01-12", "2022-01-20"])
    broken = _drop_days(df, "RegionB", days)
    with pytest.raises(ValueError) as error:
        _model().fit(broken)
    message = str(error.value)
    assert "RegionB: 4 missing" in message and "2022-01-05" in message
    assert "RegionA:" not in message and "Total:" not in message


def test_inner_keeps_the_common_dates_and_warns_with_the_rows_dropped_per_node():
    df = _data()
    broken = _drop_days(df, "RegionB", pd.to_datetime(["2022-01-05", "2022-01-09"]))
    model = _model(align="inner")
    with pytest.warns(UserWarning, match="align='inner'") as record:
        model.fit(broken)
    text = " ".join(str(w.message) for w in record if "align='inner'" in str(w.message))
    assert "Total: 2" in text and "RegionA: 2" in text and "RegionB:" not in text
    for node in model.nodes:
        assert model.sub_models[node].coefficients_ is not None


def test_inner_equals_a_strict_fit_on_the_common_dates():
    df = _data()
    days = pd.to_datetime(["2022-01-05", "2022-01-09"])
    broken = _drop_days(df, "RegionB", days)
    common = df[~df["timestamp"].isin(days)].reset_index(drop=True)
    with pytest.warns(UserWarning):
        inner = _model(align="inner").fit(broken)
    strict = _model().fit(common)
    for node in strict.nodes:
        np.testing.assert_allclose(
            inner.sub_models[node].coefficients_.detach().numpy(),
            strict.sub_models[node].coefficients_.detach().numpy(), rtol=1e-12, atol=1e-12,
        )


def test_inner_with_no_common_date_is_refused():
    df = _data()
    df.loc[df["node"] == "RegionB", "timestamp"] += pd.Timedelta(days=1000)
    with pytest.raises(ValueError, match="no (group, date)|no common"):
        _model(align="inner").fit(df)


def test_aligned_data_is_unchanged_whatever_the_row_order():
    """Aligned data gives the same fit in both modes, and shuffling the rows of a node changes nothing (nodes are matched on the date)."""
    df = _data()
    reference = _model().fit(df)
    shuffled = pd.concat([
        df[df["node"] == node].sample(frac=1.0, random_state=3) for node in ("RegionB", "Total", "RegionA")
    ], ignore_index=True)
    for model in (_model().fit(shuffled), _model(align="inner").fit(shuffled)):
        for node in reference.nodes:
            np.testing.assert_allclose(
                model.sub_models[node].coefficients_.detach().numpy(),
                reference.sub_models[node].coefficients_.detach().numpy(), rtol=1e-10, atol=1e-10,
            )


def test_strict_without_a_date_column_needs_the_same_row_count_per_node():
    df = _data().drop(columns="timestamp")
    model = HierarchicalTAM(structure=STRUCTURE, formulas=FORMULA, node_col="node")
    model.fit(df)
    short = df.drop(df[df["node"] == "RegionB"].index[:3]).reset_index(drop=True)
    with pytest.raises(ValueError, match="row count|rows"):
        HierarchicalTAM(structure=STRUCTURE, formulas=FORMULA, node_col="node").fit(short)


def test_a_non_additive_parent_is_reported_once_at_fit():
    df = _data(additive=False)
    model = _model()
    with pytest.warns(UserWarning, match="not the sum") as record:
        model.fit(df)
    assert len([w for w in record if "not the sum" in str(w.message)]) == 1
    assert "Total" in str(record[0].message)
    assert model.additivity_gap_["Total"] > 1e-3


def test_an_additive_hierarchy_is_silent_and_its_gap_is_zero():
    df = _data(additive=True)
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(df)
    assert model.additivity_gap_["Total"] < 1e-12


def test_the_additivity_gap_is_max_abs_difference_over_mean_abs_parent():
    df = _data(additive=False)
    model = _model()
    with pytest.warns(UserWarning):
        model.fit(df)
    piv = df.pivot(index="timestamp", columns="node", values="load")
    expected = (piv["Total"] - piv["RegionA"] - piv["RegionB"]).abs().max() / piv["Total"].abs().mean()
    np.testing.assert_allclose(model.additivity_gap_["Total"], expected, rtol=1e-12)
