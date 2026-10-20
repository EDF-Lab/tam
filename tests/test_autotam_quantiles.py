# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Quantile forecasts of the final model: the point forecast plus the quantiles of its validation errors.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=700, seed=2):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = 30 + 4 * x1 + 2 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(scale=1.5, size=n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="D"), "x1": x1, "y": y})


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    frame = _frame()
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=3, pop_size=6, use_opera=True, seed=6, export_dir=str(tmp_path_factory.mktemp("quantiles")))
    model.fit(frame.iloc[:450], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_the_quantile_columns_are_present_and_never_cross(fitted):
    model, frame = fitted
    out = model.predict_quantiles(frame.iloc[450:500], taus=(0.9, 0.1, 0.5), date_col="ds")
    assert list(out.columns) == ["q0.1", "q0.5", "q0.9"] and len(out) == 50
    assert (out["q0.1"] <= out["q0.5"]).all() and (out["q0.5"] <= out["q0.9"]).all()


def test_the_ten_ninety_band_covers_about_eighty_percent_of_the_test_rows(fitted):
    model, frame = fitted
    test = frame.iloc[450:]
    out = model.predict_quantiles(test, taus=(0.1, 0.9), date_col="ds")
    coverage = float(((test["y"].to_numpy() >= out["q0.1"].to_numpy()) & (test["y"].to_numpy() <= out["q0.9"].to_numpy())).mean())
    assert abs(coverage - 0.8) <= 0.08, coverage


def test_the_median_is_close_to_the_point_forecast(fitted):
    model, frame = fitted
    test = frame.iloc[450:480]
    point = model.predict(test, date_col="ds")[model.final_model_column_]
    median = model.predict_quantiles(test, taus=(0.5,), date_col="ds")["q0.5"]
    assert np.abs(median - point).max() < 1.0


@pytest.mark.parametrize("bad", [(0.0,), (1.0,), (1.2, 0.5), ()])
def test_levels_outside_zero_one_are_refused(fitted, bad):
    model, frame = fitted
    with pytest.raises(ValueError, match="taus"):
        model.predict_quantiles(frame.iloc[450:455], taus=bad, date_col="ds")


def test_a_model_without_a_final_model_raises_a_clear_error(tmp_path):
    frame = _frame(300)
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=6, export_dir=str(tmp_path))
    model.fit(frame.iloc[:250], date_col="ds", expansions=EXPANSIONS)
    with pytest.raises(ValueError, match="use_opera=True"):
        model.predict_quantiles(frame.iloc[250:260], date_col="ds")
