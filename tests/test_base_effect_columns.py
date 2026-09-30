# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""add_base_effects feeds the base model's components to AdaptiveTAM / KalmanTAM under their decomposition names."""

import warnings

import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model._math import decomposition_names


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x1": rng.normal(size=n),
                       "x5": rng.normal(size=n), "x8": rng.uniform(size=n)})
    df["y"] = np.sin(df.x5) + df.x1 * df.x8 + rng.normal(scale=0.1, size=n)
    return df


def meta(kind, base):
    if kind == "adaptive":
        return ta.AdaptiveTAM(base_model=base, adaptive_formula="y ~ l(x1)", add_base_effects=True,
                              update_interval_periods=30, training_window_periods=120, steps_per_period=1)
    return ta.KalmanTAM(base_model=base, kalman_formula="y ~ l(x1)", add_base_effects=True, date_col="date")


@pytest.mark.parametrize("kind", ["adaptive", "kalman"])
@pytest.mark.parametrize("formula", ["y ~ s(x5, k=6) + l(x5)", "y ~ te(l(x1), l(x8)) + l(x5)"])
def test_add_base_effects_uses_the_decomposition_columns(kind, formula, data):
    base = ta.StaticTAM(formula=formula, date_col="date").fit(data.iloc[:350])
    model = meta(kind, base)
    formula_out = getattr(model, "adaptive_formula_", None) or model.kalman_formula_
    for name in decomposition_names(base.effects_list_):
        assert f"l(effect_{name})" in formula_out
    columns = base.decompose_prediction(data.iloc[:5]).columns
    for term in formula_out.split("~")[1].split("+"):
        col = term.strip()[2:-1]
        if col.startswith("effect_"):
            assert col in columns, (col, list(columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.predict_online(data)
    assert len(out) > 0
