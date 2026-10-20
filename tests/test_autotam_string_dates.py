# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Dates read from a file are strings: ``fit`` turns them into timestamps, and ``predict`` of a frame whose dates are still strings must do the same
before it compares them with the training rows kept for the lags.
"""

import numpy as np
import pandas as pd

from tam.model.autotam.auto_tam import AutoTAM


def _frame(n=200):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = 10 + 2 * x + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"), "x1": x, "y": y})


def test_predict_with_string_dates_after_a_fit_with_string_dates(tmp_path):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=1, pop_size=6, use_opera=False, seed=2, lags=[7], seasonal_anchor=True, export_dir=str(tmp_path))
    model.fit(frame.iloc[:160].copy(), date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
    test = frame.iloc[160:].copy()
    assert test["ds"].map(type).eq(str).all()
    out = model.predict(test, date_col="ds")
    assert len(out) == len(test) and np.isfinite(out[model.trained_experts[0]["name"]].to_numpy()).all()
