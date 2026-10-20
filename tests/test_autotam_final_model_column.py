# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The final AutoTAM model is named, not guessed.

``predict()`` returns one column per expert (in search order), then the league ensembles, then the final model.
The first column is an expert and not the final model: ``final_model_column_`` says which column to use.
"""

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    import os
    folder = tmp_path_factory.mktemp("autotam_final_column")
    previous = os.getcwd()
    os.chdir(folder)
    try:
        rng = np.random.default_rng(42)
        n = 100
        frame = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
            "load": rng.normal(100, 10, n),
            "temp": rng.normal(20, 5, n),
            "humidity": rng.uniform(30, 90, n),
        })
        model = AutoTAM("load ~ AutoPipe(temp, humidity)", n_experts=1, pop_size=4, use_opera=True)
        model.fit(frame, date_col="ds", expansions={"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False})
        yield model, frame
    finally:
        os.chdir(previous)


def test_the_final_model_column_is_a_documented_constant():
    assert AutoTAM.FINAL_MODEL_COLUMN == "AutoTAM_Apex_Ensemble"
    assert AutoTAM.FINAL_MODEL_COLUMN_ONLINE == "AutoTAM_Apex_Online"


def test_an_unfitted_model_has_no_final_model_column():
    assert AutoTAM("load ~ AutoPipe(temp)").final_model_column_ is None


def test_fit_sets_the_final_model_column_and_predict_returns_it(fitted):
    model, frame = fitted
    assert model.final_model_column_ == AutoTAM.FINAL_MODEL_COLUMN
    predictions = model.predict(frame, date_col="ds")
    assert model.final_model_column_ in predictions.columns
    assert predictions[model.final_model_column_].notna().all()


def test_the_final_model_is_not_the_first_column(fitted):
    """Documents the reason of the attribute: the columns keep the search order, the final model comes after the experts."""
    model, frame = fitted
    columns = list(model.predict(frame, date_col="ds").columns)
    assert columns.index(model.final_model_column_) > 0


def test_predict_online_adds_the_online_final_model(fitted):
    model, frame = fitted
    online = model.predict_online(frame, date_col="ds")
    assert AutoTAM.FINAL_MODEL_COLUMN_ONLINE in online.columns


@pytest.mark.parametrize("method", ["predict", "predict_online"])
def test_the_docstrings_describe_the_columns_and_name_the_final_model(method):
    doc = getattr(AutoTAM, method).__doc__
    assert "Returns:" in doc
    assert "final model" in doc.lower()
    assert "final_model_column_" in doc or "FINAL_MODEL_COLUMN" in doc
