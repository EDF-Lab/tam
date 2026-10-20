# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Refit with the architectures of an earlier search.

One search serves many windows of a task: ``fit(..., architectures=<fitted AutoTAM>)`` skips the evolutionary search, expands the champion
formulas of the earlier search and refits them on the rows given.
"""

import time

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

EXPANSIONS = {"prior": True, "autofit": False, "kalman": False, "adaptive": False, "grid": False}


def _series(n, start=0):
    rng = np.random.default_rng(3)
    t = np.arange(start, start + n)
    x1 = rng.normal(size=n)
    y = 20 + 3 * np.sin(2 * np.pi * t / 7) + 2 * x1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D") + pd.Timedelta(days=start), "x1": x1, "y": y})


def _model(tmp_path, **kwargs):
    return AutoTAM("y ~ AutoPipe(x1)", n_experts=2, pop_size=4, use_opera=False, seed=9, export_dir=str(tmp_path), **kwargs)


@pytest.fixture(scope="module")
def searched(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("architectures")
    frame = _series(200)
    started = time.perf_counter()
    model = _model(tmp).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS)
    return model, frame, time.perf_counter() - started


def test_a_fitted_model_keeps_the_champion_formulas_of_its_search(searched):
    model, _, _ = searched
    assert model.architectures_ and all(isinstance(f, str) for formulas in model.architectures_.values() for f in formulas)


def test_the_refit_skips_the_search_and_keeps_the_experts(searched, tmp_path, caplog):
    model, frame, search_seconds = searched
    started = time.perf_counter()
    with caplog.at_level("INFO"):
        refit = _model(tmp_path).fit(frame.iloc[:180], date_col="ds", expansions=EXPANSIONS, architectures=model)
    seconds = time.perf_counter() - started
    assert "Starting Evolutionary Search" not in caplog.text and "refitting" in caplog.text
    # engineered features can differ with the rows (a purge or a window), a formula that names a feature absent now is dropped
    assert refit.architectures_ and all(f in model.architectures_.get(i, []) for i, fs in refit.architectures_.items() for f in fs)
    assert seconds < search_seconds
    assert refit.trained_experts


def test_the_refit_coefficients_follow_the_new_rows(searched, tmp_path):
    model, frame, _ = searched
    refit = _model(tmp_path).fit(frame.iloc[50:200], date_col="ds", expansions=EXPANSIONS, architectures=model)
    future = frame.iloc[190:200]
    old = model.predict(future, date_col="ds")[model.final_model_column_] if model.final_model_column_ else None
    new = refit.predict(future, date_col="ds")
    column = refit.final_model_column_ or [c for c in new.columns if c != "ds"][0]
    assert np.isfinite(new[column]).all()
    if old is not None:
        assert not np.allclose(old.to_numpy(), new[column].to_numpy())


def test_the_architectures_can_be_given_as_a_dictionary(searched, tmp_path):
    model, frame, _ = searched
    refit = _model(tmp_path).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS, architectures=dict(model.architectures_))
    assert refit.architectures_ == model.architectures_


def test_formulas_naming_a_missing_column_are_dropped_and_none_left_raises(searched, tmp_path, caplog):
    model, frame, _ = searched
    architectures = {"LinearIsland": ["y ~ l(gone)"], "Continent": model.architectures_[next(iter(model.architectures_))]}
    with caplog.at_level("WARNING"):
        refit = _model(tmp_path).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS, architectures=architectures)
    assert "dropped" in caplog.text and "LinearIsland" not in refit.architectures_
    with pytest.raises(ValueError, match="no formula"):
        _model(tmp_path).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS, architectures={"LinearIsland": ["y ~ l(gone)"]})


def test_an_unfitted_model_or_a_wrong_type_is_refused(searched, tmp_path):
    _, frame, _ = searched
    with pytest.raises(ValueError, match="not fitted"):
        _model(tmp_path).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS, architectures=_model(tmp_path))
    with pytest.raises(ValueError, match="dictionary"):
        _model(tmp_path).fit(frame.iloc[:150], date_col="ds", expansions=EXPANSIONS, architectures=["y ~ l(x1)"])
