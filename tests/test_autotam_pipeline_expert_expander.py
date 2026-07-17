# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

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
