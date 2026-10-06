# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Without date_col, the internal dummy date must cover any row count (daily spacing overflowed past ~95k rows)."""

import numpy as np
import pandas as pd

import tam as ta
from tam.common.utils import _ensure_dummies


def test_dummy_date_spans_large_inputs_in_order():
    df = pd.DataFrame({"x": np.zeros(200_000)})
    out = _ensure_dummies(df, "__dummy_group__", "__dummy_date__")
    dates = out["__dummy_date__"]
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates.iloc[-1] < pd.Timestamp("2262-01-01")


def test_static_tam_fits_without_date_col_past_the_old_bound():
    rng = np.random.default_rng(0)
    n = 120_000
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 2.0 * df["x"] + rng.normal(scale=0.1, size=n)
    model = ta.StaticTAM(formula="y ~ l(x)").fit(df)
    pred = model.predict(df.head(1000))
    assert np.isfinite(pred.iloc[:, -1].to_numpy(dtype=float)).all()
