# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
A frame whose columns are stored in another order than the formula lists them gives the same forecasts: the tensors are built from a
contiguous copy of the selected columns (a view of columns taken in reverse order has negative strides, which torch refuses).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model.neural import NeuralTAM


def _experts(n=90, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "E1": rng.normal(size=n), "E2": rng.normal(size=n), "E3": rng.normal(size=n)})
    df["y"] = 0.5 * df["E1"] + 0.3 * df["E2"] + 0.2 * df["E3"] + rng.normal(0, 0.1, n)
    return df


@pytest.mark.parametrize("columns", [["date", "y", "E1", "E2", "E3"], ["date", "y", "E3", "E2", "E1"], ["E2", "E3", "E1", "y", "date"]])
def test_opera_forecasts_do_not_depend_on_the_storage_order_of_the_expert_columns(columns):
    df = _experts()
    reference = ta.OperaTAM("y ~ l(E1) + l(E2) + l(E3)", date_col="date").predict_online(df[["date", "y", "E1", "E2", "E3"]])
    out = ta.OperaTAM("y ~ l(E1) + l(E2) + l(E3)", date_col="date").predict_online(df[columns])
    np.testing.assert_array_equal(out["prediction_opera"].to_numpy(), reference["prediction_opera"].to_numpy())


@pytest.mark.parametrize("columns", [["date", "y", "x1", "x2", "x3"], ["date", "y", "x3", "x2", "x1"]])
def test_neural_tam_does_not_depend_on_the_storage_order_of_the_columns_of_a_network(columns):
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    df["y"] = df["x1"] + df["x2"] * df["x3"] + rng.normal(0, 0.1, n)

    def forecast(frame):
        model = NeuralTAM(formula="y ~ l(x1) + n(x1, others='x2|x3', n_neurons=4)", date_col="date", epochs=3, patience=3, backfit_cycles=1, guard=False).fit(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.predict(frame)["Estimatedy"].to_numpy()

    np.testing.assert_array_equal(forecast(df[columns]), forecast(df[["date", "y", "x1", "x2", "x3"]]))
