# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for the I/O safeguards and the driver scores of
``tam.model.autotam.evaluation.autotam_report_generator``.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.evaluation.autotam_report_generator import (
    _artifact_path, _driver_scores, _safe_run_id, generate_autotam_report,
)


def test_timestamp_run_ids_are_accepted():
    assert _safe_run_id("20260915_072620") == "20260915_072620"


@pytest.mark.parametrize("run_id", ["../../secrets", "a/b", "c:\\temp", "", "run id"])
def test_run_ids_with_path_characters_are_rejected(run_id):
    with pytest.raises(ValueError):
        _safe_run_id(run_id)


def test_artifact_paths_stay_inside_the_export_directory(tmp_path):
    export_dir = tmp_path.resolve()
    assert _artifact_path(export_dir, "AutoTAM_history_1.csv").parent == export_dir
    with pytest.raises(PermissionError):
        _artifact_path(export_dir, "../outside.csv")


def test_driver_scores_use_exported_importance_and_drop_the_intercept():
    df = pd.DataFrame({"Feature_Effect": ["offset", "toy", "temperature"],
                       "Variance": [5.0e6, 2.0e6, 1.0e6], "Importance": [0.0, 0.6, 0.3]})
    drivers, label = _driver_scores(df)
    assert list(drivers["Feature_Effect"]) == ["toy", "temperature"]
    np.testing.assert_allclose(drivers["Score"], [0.6, 0.3])
    assert "Var(h_j)" in label


def test_driver_scores_fall_back_to_variance_shares_for_old_exports():
    df = pd.DataFrame({"Feature_Effect": ["toy", "temperature"], "Variance": [3.0, 1.0]})
    drivers, _ = _driver_scores(df)
    np.testing.assert_allclose(drivers["Score"], [0.75, 0.25])


def test_report_is_skipped_without_matplotlib(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    assert generate_autotam_report(str(tmp_path), run_id="20260915_072620") is None
    assert "matplotlib is required" in capsys.readouterr().out
