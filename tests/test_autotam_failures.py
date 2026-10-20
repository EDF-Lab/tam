# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Failures of the AutoTAM search are logged, never swallowed.

A candidate model that fails to fit is dropped and the search goes on, but the reason must be visible: the guard test
fails when a bare ``except Exception:`` comes back in the package, and the helper test pins the message.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tam.model.autotam as autotam_package
from tam.model.autotam.failures import log_failure
from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.pipeline import expert_expander as expander_module
from tam.model.autotam.pipeline.context import PipelineContext
from tam.model.autotam.pipeline.expert_expander import ExpertExpander

PACKAGE = Path(autotam_package.__file__).parent


def test_log_failure_names_what_failed_the_formula_and_the_error(caplog):
    logger = logging.getLogger("tam.model.autotam.test_failures")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        log_failure(logger, "KalmanTAM expert of Island_R1", ValueError("bad state"), "y ~ l(effect_x)")
    [message] = [r.getMessage() for r in caplog.records]
    assert "KalmanTAM expert of Island_R1" in message
    assert "y ~ l(effect_x)" in message
    assert "ValueError" in message and "bad state" in message


def test_log_failure_without_a_formula_has_no_formula_part(caplog):
    logger = logging.getLogger("tam.model.autotam.test_failures")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        log_failure(logger, "Date detection", KeyError("ds"))
    assert "formula" not in caplog.records[0].getMessage()


def test_log_failure_level_can_be_lowered(caplog):
    logger = logging.getLogger("tam.model.autotam.test_failures")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_failure(logger, "Correlation", ValueError("x"), level=logging.DEBUG)
    assert caplog.records[0].levelno == logging.DEBUG


def test_no_bare_except_exception_remains_in_the_package():
    """``except Exception:`` without a name hides the error: bind it and log it."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*except\s+(Exception|BaseException)\s*:", line) or re.match(r"\s*except\s*:", line):
                offenders.append(f"{path.relative_to(PACKAGE)}:{number}")
    assert not offenders, "bare except without a name: " + ", ".join(offenders)


# --------------------------------------------------------------------------- #
# Each kind of expert that fails is logged with its formula and the search goes on
# --------------------------------------------------------------------------- #

def _context(n=80):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
        "load": rng.normal(100, 10, n),
        "temp": rng.normal(20, 5, n),
    })
    return PipelineContext(
        target="load", df_fit=df.iloc[:50], df_dev=df.iloc[50:], date_col="ds",
        cv_folds=[(df.iloc[:40], df.iloc[40:50])],
        formula_config={"features": ["temp"], "targets": ["load"]},
        search_space={"temp": {"grids": {"s": {"k": [3, 5]}}}},
    )


class _Reporter:
    def export_pdp_variance(self, *args, **kwargs):
        pass


class _Boom:
    """Stands in for a model class: building it raises."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("intentional failure")


def _generate(expansions, monkeypatch, patched, caplog):
    for name in patched:
        monkeypatch.setattr(expander_module, name, _Boom)
    expander = ExpertExpander(expansions={"prior": False, "autofit": False, "kalman": False, "adaptive": False, "grid": False, **expansions})
    with caplog.at_level(logging.WARNING, logger=expander_module.logger.name):
        pool = expander.generate_experts({"Island_1": ["load ~ s(temp, k=5)"]}, _context(), [], _Reporter())
    return pool, [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failing_prior_expert_is_logged_with_its_formula(monkeypatch, caplog):
    pool, warnings = _generate({"prior": True}, monkeypatch, ["StaticTAM"], caplog)
    assert pool == {}
    assert any("Prior-based expert of Island_1_R1" in w and "load ~ s(temp, k=5)" in w and "intentional failure" in w for w in warnings)


def test_a_failing_auto_fit_expert_is_logged_with_its_formula(monkeypatch, caplog):
    pool, warnings = _generate({"autofit": True}, monkeypatch, ["StaticTAM"], caplog)
    assert any("Auto-fit expert of Island_1_R1" in w and "load ~ s(temp, k=5)" in w for w in warnings)


def test_a_failing_grid_expert_is_logged_with_its_formula(monkeypatch, caplog):
    pool, warnings = _generate({"grid": True}, monkeypatch, ["StaticTAM"], caplog)
    assert any("Grid-search expert of Island_1_R1" in w and "load ~ s(temp, k=5)" in w for w in warnings)


def test_a_failing_kalman_expert_is_logged_and_the_static_expert_stays(monkeypatch, caplog):
    pool, warnings = _generate({"prior": True, "kalman": True}, monkeypatch, ["KalmanTAM"], caplog)
    assert any(name.startswith("Static_") for name in pool)
    assert not any(name.startswith("Kalman_") for name in pool)
    kalman_warnings = [w for w in warnings if "KalmanTAM expert" in w]
    assert kalman_warnings and all("intentional failure" in w and "l(effect_" in w for w in kalman_warnings)


def test_a_failing_adaptive_expert_is_logged_and_the_static_expert_stays(monkeypatch, caplog):
    pool, warnings = _generate({"prior": True, "adaptive": True}, monkeypatch, ["AdaptiveTAM"], caplog)
    assert any(name.startswith("Static_") for name in pool)
    assert any("AdaptiveTAM expert" in w and "intentional failure" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Pruning: a candidate whose fit failed has nothing to prune and is not a new failure
# --------------------------------------------------------------------------- #

def _prune(model, caplog):
    terms = [{"feature": "temp", "type": "l", "params": {}}]
    frame = pd.DataFrame({"load": [1.0, 2.0, 3.0], "temp": [1.0, 2.0, 3.0]})
    with caplog.at_level(logging.DEBUG):
        kept = KnowledgeGraph().update_and_prune(terms, model, frame, "load", global_rmse=1.0, target_std=1.0)
    return terms, kept


def test_a_candidate_without_a_fitted_model_is_not_pruned_and_logs_nothing(caplog):
    terms, kept = _prune(None, caplog)
    assert kept == terms
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_model_that_cannot_decompose_is_logged_once_and_not_pruned(caplog):
    class _Broken:
        def decompose_prediction(self, df):
            raise RuntimeError("cannot decompose")

    terms, kept = _prune(_Broken(), caplog)
    assert kept == terms
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "cannot decompose" in warnings[0]

