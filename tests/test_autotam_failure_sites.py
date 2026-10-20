# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
One test per failure site of the AutoTAM package that logs and goes on: each makes its call raise, and checks the log record (what failed, the
error) and the documented fall-back. A test fails if its ``log_failure`` call is removed.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam import drag_tam as drag_module
from tam.model.autotam.auto_tam import AutoTAM
from tam.model.autotam.drag_tam import DragTAM
from tam.model.autotam.evolution_reporter import EvolutionReporter
from tam.model.autotam.feature_engineer import FeatureEngineer
from tam.model.autotam.knowledge_graph import KnowledgeGraph
from tam.model.autotam.pipeline import ensemble_selector as selector_module
from tam.model.autotam.pipeline.data_manager import DataManager

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _frame(n=140):
    rng = np.random.default_rng(5)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "x1": x1, "y": 10 + 2 * x1 + rng.normal(scale=0.3, size=n)})


def _logged(caplog, text, level=None):
    records = [r for r in caplog.records if text in r.getMessage() and (level is None or r.levelno == level)]
    assert records, f"no log record with {text!r}: {[r.getMessage()[:80] for r in caplog.records]}"
    return records[0]


@pytest.fixture(scope="module")
def survived(tmp_path_factory):
    """A fit during which the clipping bounds of the selector and the PDP export fail: the fit goes on."""
    mp = pytest.MonkeyPatch()
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    logging.getLogger("tam.model.autotam").addHandler(handler)
    logging.getLogger("tam.model.autotam").setLevel(logging.DEBUG)
    mp.setattr(selector_module, "guard_bounds", lambda low, high: (_ for _ in ()).throw(RuntimeError("bounds broke")))
    mp.setattr(EvolutionReporter, "export_pdp_variance", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("pdp broke")))
    try:
        model = AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=True, seed=2, extrapolation_guards=True,
                        export_dir=str(tmp_path_factory.mktemp("sites")))
        model.fit(_frame(), date_col="ds", expansions=EXPANSIONS)
    finally:
        mp.undo()
        logging.getLogger("tam.model.autotam").removeHandler(handler)
    return model, records


def _find(records, text):
    found = [r for r in records if text in r.getMessage()]
    assert found, f"no log record with {text!r}"
    return found[0]


def test_the_clipping_bounds_of_the_selector_fall_back_to_no_clipping_and_the_fit_goes_on(survived):
    model, records = survived
    record = _find(records, "Training bounds of the target for the clipping of the experts")
    assert record.levelno == logging.WARNING and "bounds broke" in record.getMessage()
    assert model.ctx.target_clip is None and model.trained_experts


def test_the_pdp_export_failure_is_logged_and_the_fit_goes_on(survived):
    _, records = survived
    record = _find(records, "PDP export of the global static champion")
    assert record.levelno == logging.WARNING and "pdp broke" in record.getMessage()


def test_the_clipping_bounds_of_predict_fall_back_to_no_clipping(survived, caplog):
    model, _ = survived
    frame = _frame()

    class Broken:
        def __getitem__(self, key):
            raise RuntimeError("no fit rows")

    model.ctx.df_fit = Broken()
    model.ctx.target_clip = None
    with caplog.at_level(logging.WARNING):
        out = model.predict(frame.iloc[100:105], date_col="ds")
    record = _logged(caplog, "Training bounds of the target for the clipping of the experts", logging.WARNING)
    assert "no fit rows" in record.getMessage() and len(out) == 5


def test_the_alignment_of_the_predictions_falls_back_to_positions(survived, caplog):
    model, _ = survived
    source = pd.DataFrame({"ds": ["not a date"] * 3, "y": [1.0, 2.0, 3.0]})
    clean = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3), "y": [np.nan] * 3})
    model.ctx.date_col, model.ctx.group_col = "ds", None
    with caplog.at_level(logging.WARNING):
        out = model._align_prediction_to_test(pd.Series([1.0, 2.0, 3.0]), source, clean)
    record = _logged(caplog, "Alignment of the predictions on the key columns", logging.WARNING)
    assert "falls back to positions" in record.getMessage()
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_the_alignment_of_the_test_frame_falls_back_to_the_tail_rows(survived, caplog, monkeypatch):
    model, _ = survived
    frame = _frame().iloc[120:125]
    monkeypatch.setattr(pd.DataFrame, "merge", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("merge broke")))
    with caplog.at_level(logging.WARNING):
        clean, augmented = model.data_manager.transform_test_data(frame, model.ctx)
    record = _logged(caplog, "Alignment of the test frame on the augmented frame", logging.WARNING)
    assert "merge broke" in record.getMessage() and len(clean) == len(frame)


def test_the_date_detection_of_the_validation_split_falls_back_to_a_non_time_series_split(caplog):
    frame = _frame(80).assign(ds=lambda d: ["not a date"] * len(d))
    manager = DataManager("y ~ AutoPipe(x1)")
    with caplog.at_level(logging.WARNING):
        folds = manager.generate_cv_folds(frame.iloc[:56], frame.iloc[56:], "ds", "auto")
    record = _logged(caplog, "Date and frequency detection of the validation split", logging.WARNING)
    assert "non-time-series" in record.getMessage() and folds


def _drag():
    return DragTAM(target_col="y", population_size=4, group_col=None, date_col=None)


def test_a_cross_validation_fold_that_fails_is_logged_at_debug_and_the_score_is_infinite(caplog):
    frame = _frame()
    fold = (frame.iloc[:100], frame.iloc[100:])
    with caplog.at_level(logging.DEBUG, logger="tam.model.autotam.drag_tam"):
        score, model, terms, penalties = _drag()._evaluate_candidate("y ~ l(missing_column)", [fold])
    record = _logged(caplog, "Cross-validation fold of a candidate formula", logging.DEBUG)
    assert "y ~ l(missing_column)" in record.getMessage()
    assert score == float("inf") and model is None


def test_a_final_fit_that_fails_is_logged_at_debug_and_the_score_is_infinite(caplog, monkeypatch):
    calls = {"n": 0}

    class Stub:
        def __init__(self, *a, **k):
            pass

        def fit(self, df):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("final fit broke")

        def predict(self, df):
            return pd.DataFrame({"Estimatedy": df["y"].to_numpy()})

    monkeypatch.setattr(drag_module, "StaticTAM", Stub)
    frame = _frame()
    with caplog.at_level(logging.DEBUG, logger="tam.model.autotam.drag_tam"):
        score, model, terms, penalties = _drag()._evaluate_candidate("y ~ l(x1)", [(frame.iloc[:100], frame.iloc[100:])])
    record = _logged(caplog, "Final fit of a candidate formula", logging.DEBUG)
    assert "final fit broke" in record.getMessage() and score == float("inf")


def test_the_correlation_with_the_calendar_signals_failing_is_logged_at_debug(caplog):
    frame = _frame().assign(ds=lambda d: ["not a date"] * len(d))
    with caplog.at_level(logging.DEBUG, logger="tam.model.autotam.feature_engineer"):
        FeatureEngineer()._detect_protected_features(frame, "ds", "y")
    record = _logged(caplog, "Correlation of a column with the target signal", logging.DEBUG)
    assert record.levelno == logging.DEBUG


def test_the_detection_of_a_column_copied_from_the_target_failing_is_logged_at_debug(caplog):
    frame = _frame().assign(y=lambda d: ["a"] * len(d))
    with caplog.at_level(logging.DEBUG, logger="tam.model.autotam.feature_engineer"):
        FeatureEngineer()._detect_protected_features(frame, None, "y")
    _logged(caplog, "Detection of a column copied from the target", logging.DEBUG)


def test_a_model_that_cannot_decompose_leaves_the_terms_unpruned(caplog):
    class Broken:
        def decompose_prediction(self, df):
            raise RuntimeError("decompose broke")

    terms = [{"type": "l", "feature": "x1", "params": {}}]
    with caplog.at_level(logging.WARNING):
        kept = KnowledgeGraph().update_and_prune(terms, Broken(), _frame(), "y", global_rmse=1.0, target_std=1.0)
    record = _logged(caplog, "Contribution of the terms to the prediction variance", logging.WARNING)
    assert "decompose broke" in record.getMessage() and kept == terms
