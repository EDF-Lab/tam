# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Quantile forecasts of the final model aggregated level by level: the quantile forecast of each member of the final model, weighted by
one ``OperaTAM`` with the pinball loss per level.
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
    model = AutoTAM("y ~ AutoPipe(x1)", n_experts=3, pop_size=6, use_opera=True, seed=6, export_dir=str(tmp_path_factory.mktemp("opera_q")))
    model.fit(frame.iloc[:450], date_col="ds", expansions=EXPANSIONS)
    return model, frame


def test_the_levels_are_aggregated_and_never_cross(fitted):
    model, frame = fitted
    out = model.predict_quantiles(frame.iloc[450:500], taus=(0.9, 0.1, 0.5), date_col="ds", aggregation="opera")
    assert list(out.columns) == ["q0.1", "q0.5", "q0.9"] and len(out) == 50
    assert np.isfinite(out.to_numpy()).all()
    assert (out["q0.1"] <= out["q0.5"]).all() and (out["q0.5"] <= out["q0.9"]).all()


def test_the_aggregated_band_covers_about_eighty_percent_of_the_test_rows(fitted):
    model, frame = fitted
    test = frame.iloc[450:]
    out = model.predict_quantiles(test, taus=(0.1, 0.9), date_col="ds", aggregation="opera")
    coverage = float(((test["y"].to_numpy() >= out["q0.1"].to_numpy()) & (test["y"].to_numpy() <= out["q0.9"].to_numpy())).mean())
    assert abs(coverage - 0.8) <= 0.1, coverage


def test_the_empirical_forecast_is_the_default(fitted):
    model, frame = fitted
    test = frame.iloc[450:470]
    pd.testing.assert_frame_equal(model.predict_quantiles(test, date_col="ds"), model.predict_quantiles(test, date_col="ds", aggregation="empirical"))


def test_with_one_member_the_aggregation_is_the_empirical_forecast(fitted, monkeypatch):
    model, frame = fitted
    test = frame.iloc[450:470]
    monkeypatch.setattr(model, "weights_top10", {next(iter(model.weights_top10)): 1.0})
    pd.testing.assert_frame_equal(model.predict_quantiles(test, date_col="ds", aggregation="opera"), model.predict_quantiles(test, date_col="ds"))


def test_an_unknown_aggregation_is_refused(fitted):
    model, frame = fitted
    with pytest.raises(ValueError, match="aggregation"):
        model.predict_quantiles(frame.iloc[450:455], date_col="ds", aggregation="mean")


def test_a_model_fitted_on_increments_is_refused(fitted, monkeypatch):
    model, frame = fitted
    monkeypatch.setattr(model, "difference_", 1)
    with pytest.raises(NotImplementedError, match="increments"):
        model.predict_quantiles(frame.iloc[450:455], date_col="ds", aggregation="opera")
