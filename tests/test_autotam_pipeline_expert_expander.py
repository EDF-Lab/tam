# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Unit tests for ``tam.model.autotam.pipeline.expert_expander.ExpertExpander``.

Tests the metric calculator and the CV evaluation helper. The full
generate_experts() pipeline is not exercised here because it depends on
a complete AutoTAM search run (per the NEWTODO.md guardrail).
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.pipeline.expert_expander import ExpertExpander
from tam.model.autotam.pipeline.context import PipelineContext


def _exp() -> ExpertExpander:
    return ExpertExpander()


# --------------------------------------------------------------------------- #
# _calculate_error (identical contract to EnsembleSelector)
# --------------------------------------------------------------------------- #

def test_calculate_error_rmse():
    assert np.isclose(_exp()._calculate_error(
        np.array([10.0, 20.0, 30.0]), np.array([11.0, 19.0, 31.0]), "rmse"
    ), 1.0)


def test_calculate_error_mae():
    assert np.isclose(_exp()._calculate_error(
        np.array([10.0, 20.0, 30.0]), np.array([11.0, 19.0, 31.0]), "mae"
    ), 1.0)


def test_calculate_error_mape():
    assert np.isclose(_exp()._calculate_error(
        np.array([100.0, 200.0]), np.array([110.0, 180.0]), "mape"
    ), 10.0)


def test_calculate_error_all_nan_returns_inf():
    assert _exp()._calculate_error(
        np.array([np.nan]), np.array([1.0]), "rmse"
    ) == float("inf")


def test_calculate_error_unknown_metric_falls_back_to_rmse():
    result = _exp()._calculate_error(
        np.array([10.0, 20.0]), np.array([11.0, 19.0]), "smape"
    )
    assert np.isfinite(result)


# --------------------------------------------------------------------------- #
# _evaluate_model_cv with a mock model
# --------------------------------------------------------------------------- #

class _PerfectModel:
    """A model whose predict() returns the true target exactly."""
    def predict(self, df):
        return pd.DataFrame({"Estimatedload": df["load"].values})


class _CrashModel:
    """A model whose predict() always raises."""
    def predict(self, df):
        raise RuntimeError("intentional failure")


def test_evaluate_model_cv_perfect_model_returns_zero():
    rng = np.random.default_rng(0)
    folds = []
    for _ in range(3):
        train = pd.DataFrame({"load": rng.normal(100, 5, 20)})
        val = pd.DataFrame({"load": rng.normal(100, 5, 10)})
        folds.append((train, val))

    score = ExpertExpander()._evaluate_model_cv(_PerfectModel(), folds, "load")
    assert np.isclose(score, 0.0)


def test_evaluate_model_cv_crashing_model_returns_inf():
    folds = [(pd.DataFrame({"load": [1.0]}), pd.DataFrame({"load": [1.0]}))]
    score = ExpertExpander()._evaluate_model_cv(_CrashModel(), folds, "load")
    assert score == float("inf")


def test_evaluate_model_cv_empty_folds_returns_inf():
    score = ExpertExpander()._evaluate_model_cv(_PerfectModel(), [], "load")
    assert score == float("inf")


def test_generate_experts_grid_retains_external_mandatory_terms_verbatim():
    expander = ExpertExpander(expansions={"prior": False, "autofit": False, "kalman": False, "adaptive": False, "grid": True})

    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
        "humidity": rng.uniform(30, 90, n),
    })

    ctx = PipelineContext(
        target="load",
        df_fit=df.iloc[:40],
        df_dev=df.iloc[40:],
        cv_folds=[(df.iloc[:30], df.iloc[30:40])],
        formula_config={"features": ["temp"], "targets": ["load"]},
        search_space={"temp": {"grids": {"s": {"k": [3, 5]}}}},
        external_mandatory_features={"humidity"},
    )

    island_champions = {
        "Island_1": ["load ~ s(temp, k=3) + s(humidity, k=5)"]
    }

    test_log = []
    class DummyReporter:
        def export_pdp_variance(self, *args, **kwargs): pass

    candidates = expander.generate_experts(island_champions, ctx, test_log, DummyReporter())

    grid_candidates = [k for k in candidates.keys() if "Grid" in k]
    assert len(grid_candidates) > 0

    grid_logs = [entry for entry in test_log if entry.get("Model_Type") == "StaticTAM_grid"]
    assert len(grid_logs) > 0
    tokenized_form = grid_logs[0]["Formula"]
    assert "s(humidity, k=5)" in tokenized_form
    assert "grid_k_temp_s" in tokenized_form
    assert "humidity" not in grid_logs[0]["Hyperparameters"]


def test_generate_experts_grid_preserves_tensor_terms():
    expander = ExpertExpander(expansions={"prior": False, "autofit": False, "kalman": False, "adaptive": False, "grid": True})

    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
        "humidity": rng.uniform(30, 90, n),
    })

    ctx = PipelineContext(
        target="load",
        df_fit=df.iloc[:40],
        df_dev=df.iloc[40:],
        cv_folds=[(df.iloc[:30], df.iloc[30:40])],
        formula_config={"features": ["temp"], "targets": ["load"]},
        search_space={"temp": {"grids": {"s": {"k": [3, 5]}}}},
        external_mandatory_features={"humidity"},
    )

    island_champions = {
        "Island_1": ["load ~ s(temp, k=3) + te(s(temp), s(humidity, k=3))"]
    }

    test_log = []
    class DummyReporter:
        def export_pdp_variance(self, *args, **kwargs): pass

    candidates = expander.generate_experts(island_champions, ctx, test_log, DummyReporter())

    grid_logs = [entry for entry in test_log if entry.get("Model_Type") == "StaticTAM_grid"]
    assert len(grid_logs) > 0
    tokenized_form = grid_logs[0]["Formula"]
    assert "te(" in tokenized_form
    assert "interaction" not in tokenized_form

