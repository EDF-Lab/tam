# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for the data tensorization layer in ``tam.model._data``.

Covers per-group normalization, stacking DataFrames into 3D tensors, and
reassembling tensor outputs back into DataFrames (predictions and decomposed
effect columns).
"""

import numpy as np
import pandas as pd
import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model._data import (
    _fit_normalization_params,
    normalize,
    _transform_data_stacked,
    _reassemble_predictions,
    _reassemble_decomposed_predictions,
)


def _panel():
    """Two balanced groups, 5 rows each, single feature + target."""
    return pd.DataFrame({
        "gid": ["A"] * 5 + ["B"] * 5,
        "temp": [0.0, 10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0, 45.0],
        "load": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    })


def test_fit_normalization_params_per_group():
    params, groups = _fit_normalization_params(_panel(), features=["temp"], group_col="gid")
    assert groups == ["A", "B"]
    assert params["A"]["min"]["temp"] == 0.0
    assert params["A"]["max"]["temp"] == 40.0
    assert params["B"]["min"]["temp"] == 5.0
    assert params["B"]["max"]["temp"] == 45.0


def test_normalize_maps_to_unit_interval():
    df = pd.DataFrame({"temp": [0.0, 20.0, 40.0]})
    params = {"min": df.min(), "max": df.max()}
    out = normalize(df, params)
    # Min -> -1, midpoint -> 0, max -> +1.
    assert np.isclose(out["temp"].iloc[0], -1.0)
    assert np.isclose(out["temp"].iloc[1], 0.0)
    assert np.isclose(out["temp"].iloc[2], 1.0)


def test_normalize_handles_constant_feature():
    df = pd.DataFrame({"temp": [7.0, 7.0, 7.0]})
    params = {"min": df.min(), "max": df.max()}
    out = normalize(df, params)
    # Amplitude 0 must not divide-by-zero; centered constant -> 0.
    assert np.isfinite(out["temp"]).all()
    assert np.allclose(out["temp"].values, 0.0)


def test_transform_data_stacked_shapes():
    df = _panel()
    params, groups = _fit_normalization_params(df, features=["temp"], group_col="gid")
    x, y = _transform_data_stacked(df, ["temp"], "gid", params, groups, target_col="load")

    assert x.shape == (2, 5, 1)  # (groups, samples, features)
    assert y.shape == (2, 5, 1)
    # Each group's normalized feature spans [-1, 1].
    assert torch.isclose(x.min(), torch.tensor(-1.0, dtype=x.dtype))
    assert torch.isclose(x.max(), torch.tensor(1.0, dtype=x.dtype))


def test_transform_data_stacked_without_target():
    df = _panel()
    params, groups = _fit_normalization_params(df, features=["temp"], group_col="gid")
    x, y = _transform_data_stacked(df, ["temp"], "gid", params, groups, target_col=None)
    assert x.shape == (2, 5, 1)
    assert y is None


def test_transform_data_stacked_empty_returns_empty_tensors():
    df = _panel()
    params, groups = _fit_normalization_params(df, features=["temp"], group_col="gid")
    empty = df.iloc[0:0]
    x, y = _transform_data_stacked(empty, ["temp"], "gid", params, groups, target_col="load")
    assert x.shape[0] == 0
    assert y is not None and y.shape[0] == 0


def test_reassemble_predictions_adds_estimated_column():
    df = _panel().reset_index(drop=True)
    preds = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0],
                          [20.0, 21.0, 22.0, 23.0, 24.0]], dtype=torch.get_default_dtype())
    out = _reassemble_predictions(df, preds, "gid", ["A", "B"], target_col="load")

    assert "Estimatedload" in out.columns
    assert len(out) == len(df)
    assert not out["Estimatedload"].isna().any()
    # Group A's first prediction lands on group A's first row.
    assert out.loc[df["gid"] == "A", "Estimatedload"].iloc[0] == 10.0


def test_reassemble_decomposed_predictions_adds_effect_columns():
    df = _panel().reset_index(drop=True)
    decomposed = {
        "temp": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=torch.get_default_dtype()),
    }
    out = _reassemble_decomposed_predictions(df, decomposed, "gid", ["A", "B"])
    assert "effect_temp" in out.columns
    assert out.loc[df["gid"] == "A", "effect_temp"].iloc[0] == 1.0
    assert out.loc[df["gid"] == "B", "effect_temp"].iloc[0] == 2.0


def test_transform_data_stacked_respects_date_sort():
    """Verifies that the stacking layer chronologically sorts the data, neutralizing row-scrambling bugs."""
    df_shuffled = pd.DataFrame({
        "gid": ["A", "A", "A"],
        "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
        "temp": [30.0, 10.0, 20.0],
    })
    params, groups = _fit_normalization_params(df_shuffled, features=["temp"], group_col="gid")
    
    x, _ = _transform_data_stacked(df_shuffled, ["temp"], "gid", params, groups, date_col="date")

    assert np.isclose(x[0, 0, 0].item(), -1.0) # Jan 1
    assert np.isclose(x[0, 1, 0].item(), 0.0)  # Jan 2
    assert np.isclose(x[0, 2, 0].item(), 1.0)  # Jan 3


def test_reassemble_safely_aligns_shuffled_index():
    """Verifies that chronologically ordered tensors correctly map back to shuffled DataFrame indices."""
    df_shuffled = pd.DataFrame({
        "gid": ["A", "A", "A"],
        "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
        "load": [0.0, 0.0, 0.0]
    }, index=[99, 42, 7]) 

    preds_chronological = torch.tensor([[100.0, 200.0, 300.0]], dtype=torch.get_default_dtype())
    
    out = _reassemble_predictions(
        original_data=df_shuffled, 
        predictions_stacked=preds_chronological, 
        group_col="gid", 
        unique_groups=["A"], 
        target_col="load", 
        date_col="date"
    )
    
    assert out.loc[out["date"] == pd.to_datetime("2020-01-01"), "Estimatedload"].iloc[0] == 100.0
    assert out.loc[out["date"] == pd.to_datetime("2020-01-02"), "Estimatedload"].iloc[0] == 200.0
    assert out.loc[out["date"] == pd.to_datetime("2020-01-03"), "Estimatedload"].iloc[0] == 300.0