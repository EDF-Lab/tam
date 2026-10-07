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

import logging

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.pipeline.expert_expander import ExpertExpander, kalman_formula_from_effects
from tam.model.additive import StaticTAM
from tam.model.kalman import KalmanTAM
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


def test_evaluate_model_cv_failure_is_logged_with_formula_and_fold(caplog):
    class _NamedCrashModel(_CrashModel):
        formula_ = "load ~ s(temp, k=10)"

    folds = [(pd.DataFrame({"load": [1.0]}), pd.DataFrame({"load": [1.0]}))] * 2
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam.pipeline.expert_expander"):
        score = ExpertExpander()._evaluate_model_cv(_NamedCrashModel(), folds, "load")

    assert score == float("inf")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    for fold_index, message in enumerate(warnings):
        assert f"fold {fold_index}" in message
        assert "load ~ s(temp, k=10)" in message
        assert "intentional failure" in message


def test_evaluate_model_cv_success_logs_nothing(caplog):
    folds = [(pd.DataFrame({"load": [1.0]}), pd.DataFrame({"load": [1.0, 2.0]}))]
    with caplog.at_level(logging.WARNING, logger="tam.model.autotam.pipeline.expert_expander"):
        ExpertExpander()._evaluate_model_cv(_PerfectModel(), folds, "load")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


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


# --------------------------------------------------------------------------- #
# Kalman experts: the state formula and the search
# --------------------------------------------------------------------------- #

def _static_model_and_frame(n=120):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    frame["y"] = 2.0 * frame["x1"] - frame["x2"] + rng.normal(scale=0.1, size=n)
    model = StaticTAM(formula="y ~ l(x1) + l(x2)", date_col="date")
    model.fit(frame)
    return model, frame


def test_kalman_formula_has_one_linear_term_per_effect_column():
    assert kalman_formula_from_effects("y", ["effect_offset", "effect_x1"]) == "y ~ l(effect_offset) + l(effect_x1)"


@pytest.mark.filterwarnings("ignore:KalmanTAM:UserWarning")
@pytest.mark.filterwarnings("ignore:The calibration period includes:UserWarning")
def test_kalman_formula_built_from_the_effects_of_a_base_model_is_accepted_by_kalman_tam():
    model, frame = _static_model_and_frame()
    effects = [c for c in model.decompose_prediction(frame).columns if c.startswith("effect_")]
    assert effects, "the base model exposes no effect column"

    formula = kalman_formula_from_effects("y", effects)
    KalmanTAM(base_model=model, kalman_formula=formula, date_col="date", horizon_steps=1)

    bare_names = "y ~ " + " + ".join(effects) + " - 1"
    with pytest.raises(ValueError, match="must be function calls"):
        KalmanTAM(base_model=model, kalman_formula=bare_names, date_col="date", horizon_steps=1)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_the_search_builds_kalman_experts(tmp_path, monkeypatch):
    """Every Kalman candidate used to fail on its formula and was dropped without a trace."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    n = 100
    frame = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
        "humidity": rng.uniform(30, 90, n),
    })
    search = AutoTAM("load ~ AutoPipe(temp, humidity)", n_experts=1, pop_size=4, use_opera=False)
    search.fit(frame, date_col="ds",
               expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": False, "grid": False})

    kalman_candidates = [e for e in search.chronological_test_log if e.get("Model_Type") == "KalmanTAM"]
    assert kalman_candidates, "no KalmanTAM candidate was built by the search"
