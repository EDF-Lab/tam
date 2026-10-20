# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Dotted column names (``day.of.year``, as R and many CSV files write them) are one name everywhere in AutoTAM: cut at the dots, a candidate that uses
the column was fitted without it, failed, and the search silently dropped the feature.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.tokens import first_arguments, identifier_tokens


def test_a_dotted_name_is_one_token():
    tokens = identifier_tokens("y ~ s(day.of.year, ap=-5.0, deg=3, k=10) + l(time)")
    assert "day.of.year" in tokens and "day" not in tokens and "of" not in tokens and "time" in tokens
    assert identifier_tokens("c(x1) + l(y_lag_7)") == ["c", "x1", "l", "y_lag_7"]
    assert "x.one" in first_arguments("te(s(x.one, k=5), s(x.two))") and "x.two" in first_arguments("te(s(x.one, k=5), s(x.two))")


def _frame(n=240, seed=3):
    rng = np.random.default_rng(seed)
    one, two = rng.uniform(0, 1, n), rng.normal(size=n)
    y = 5 + 4 * np.sin(2 * np.pi * one) + 0.3 * two + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="D"), "x.one": one, "x.two": two, "y": y})


def test_the_dotted_columns_survive_the_search_and_the_forecast(tmp_path):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x.one, x.two)", n_experts=2, pop_size=8, use_opera=False, seed=3, export_dir=str(tmp_path))
    model.fit(frame.iloc[:200], date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    formulas = " ".join(f for group in model.architectures_.values() for f in group)
    assert "x.one" in formulas, formulas
    out = model.predict(frame.iloc[200:], date_col="ds")
    prediction = out[model.trained_experts[0]["name"]].to_numpy()
    assert np.isfinite(prediction).all()
    assert np.sqrt(np.mean((prediction - frame["y"].iloc[200:].to_numpy()) ** 2)) < 1.5       # the sine of x.one is found, not a flat line
