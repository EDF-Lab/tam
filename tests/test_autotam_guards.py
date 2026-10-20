# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Extrapolation guards (opt-in): ``AutoTAM(..., extrapolation_guards=True)``.

The worst forecasts of a search are extrapolations: a linear term on a counter beyond the history, a neural or PID term fitted on a few hundred rows,
a formula with as many coefficients as rows. The guards keep such candidates out of the champions, floor a never-negative target at zero, and count
the rows each expert had to be clipped on, against it in the selection. Without the argument nothing changes.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import guards
from tam.model.autotam.auto_tam import AutoTAM


def test_the_constructor_has_a_defaulted_extrapolation_guards_argument():
    parameter = inspect.signature(AutoTAM.__init__).parameters["extrapolation_guards"]
    assert parameter.default is False


# --------------------------------------------------------------------------- #
# Pure rules
# --------------------------------------------------------------------------- #

def test_the_bounds_keep_the_margin_and_floor_a_never_negative_target_at_zero():
    low, high = guards.guard_bounds(0.1, 1.0, margin_fraction=0.15)
    assert low == 0.0 and high == pytest.approx(1.135)
    low, high = guards.guard_bounds(-5.0, 5.0, margin_fraction=0.15)
    assert low == pytest.approx(-6.5) and high == pytest.approx(6.5)


def test_monotone_features_are_the_counters():
    n = 100
    frame = pd.DataFrame({"trend": np.arange(n), "back": -np.arange(n), "noise": np.random.default_rng(0).normal(size=n),
                          "season": np.sin(np.arange(n) / 5)})
    assert guards.monotone_features(frame, list(frame.columns)) == {"trend", "back"}


def _violation(formula, n_rows=100, monotone=frozenset()):
    from tam.model.autotam.pipeline.context import PipelineContext
    return guards.guard_violation(formula, n_rows, set(monotone), PipelineContext().estimate_complexity)


def test_a_clean_formula_has_no_violation():
    assert _violation("y ~ l(x1) + s(x2, k=5)") is None


def test_a_formula_with_more_coefficients_than_the_bound_is_rejected():
    reason = _violation("y ~ s(x1, k=40) + s(x2, k=40)", n_rows=100)
    assert reason is not None and "coefficients" in reason


def test_neural_and_pid_terms_need_enough_rows():
    assert "rows" in _violation("y ~ n(x1, n_neurons=4, n_hidden=1)", n_rows=300)
    assert "rows" in _violation("y ~ pid(x1, w=5)", n_rows=300)
    assert _violation("y ~ pid(x1, w=5)", n_rows=5000) is None


def test_a_linear_term_on_a_counter_is_rejected_but_a_spline_is_not():
    assert "trend" in _violation("y ~ l(trend)", monotone={"trend"})
    assert _violation("y ~ s(trend, k=5)", monotone={"trend"}) is None
    assert _violation("y ~ l(x1)", monotone={"trend"}) is None


# --------------------------------------------------------------------------- #
# In the search
# --------------------------------------------------------------------------- #

def _series(n=100, upward=False):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": rng.normal(size=n)})
    df["trend"] = np.arange(n, dtype=float)
    df["y"] = 20.0 + 2.0 * df["x1"] + (0.3 * df["trend"] if upward else 0.0) + rng.normal(scale=0.5, size=n)
    return df


EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


@pytest.fixture(scope="module")
def guarded(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_guards"))
    try:
        df = _series(upward=True)
        past, future = df.iloc[:70].copy(), df.iloc[70:].copy()
        model = AutoTAM("y ~ AutoPipe(x1, trend)", n_experts=1, pop_size=6, use_opera=False, extrapolation_guards=True)
        model.fit(past, date_col="ds", expansions=EXPANSIONS)
        yield model, past, future
    finally:
        os.chdir(previous)


def test_no_formula_of_the_search_exceeds_the_coefficient_bound(guarded):
    model, past, _ = guarded
    bound = guards.GUARD_MAX_COEFFICIENT_FRACTION * len(model.ctx.df_fit)
    formulas = [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]
    assert formulas
    assert all(model.ctx.estimate_complexity(f) <= bound for f in formulas)


def test_no_champion_has_a_linear_term_on_the_counter(guarded):
    model, _, _ = guarded
    formulas = [e["Formula"] for e in model.chronological_test_log if str(e.get("Model_Type", "")).startswith("Static")]
    assert not [f for f in formulas if "l(trend" in f.replace(" ", "")]


def test_the_forecast_stays_inside_the_guard_and_the_clipped_rows_are_reported(guarded):
    model, past, future = guarded
    far = future.copy()
    far["trend"] = far["trend"] + 1000.0  # a counter far beyond the history
    out = model.predict(far, date_col="ds")
    low, high = guards.guard_bounds(float(model.ctx.df_fit["y"].min()), float(model.ctx.df_fit["y"].max()))
    assert out.to_numpy().min() >= low - 1e-9 and out.to_numpy().max() <= high + 1e-9
    assert isinstance(model.clipped_rows_, dict)


def test_without_the_guards_nothing_is_counted_or_filtered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = AutoTAM("y ~ AutoPipe(x1, trend)", n_experts=1, pop_size=4, use_opera=False)
    model.fit(_series(n=60).iloc[:50].copy(), date_col="ds", expansions=EXPANSIONS)
    assert model.extrapolation_guards is False
    assert model.ctx.guard_rejections == {}
    assert model.clipped_rows_ == {}
