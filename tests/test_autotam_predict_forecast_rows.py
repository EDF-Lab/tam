# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``AutoTAM.predict()`` on the forecast rows alone.

The adaptive and Kalman experts need the history that precedes the rows they forecast. ``fit()`` keeps the training rows,
and ``predict()`` puts them in front of a frame that starts after them, so the forecast of a row does not depend on
whether the caller remembered to pass the past: it equals the forecast of the same row inside past + future.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

FINAL = AutoTAM.FINAL_MODEL_COLUMN


def _series(n=260):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "temp": rng.normal(20, 5, n)})
    df["load"] = 100 + 2 * df["temp"] + 10 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 2, n)
    return df


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("autotam_forecast_rows"))
    try:
        df = _series()
        past, future = df.iloc[:230].copy(), df.iloc[230:].copy()
        model = AutoTAM("load ~ AutoPipe(temp)", n_experts=1, pop_size=4, use_opera=True)
        model.fit(past, date_col="ds",
                  expansions={"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False})
        yield model, past, future
    finally:
        os.chdir(previous)


def test_an_unfitted_model_has_no_history():
    assert AutoTAM("load ~ AutoPipe(temp)").history_ is None


def test_fit_keeps_the_training_rows(fitted):
    model, past, _ = fitted
    assert len(model.history_) == len(past)
    assert list(model.history_["ds"]) == list(past["ds"])


def test_the_experts_are_dynamic_so_the_test_covers_the_history(fitted):
    model, _, _ = fitted
    kinds = {expert["type"] for expert in model.trained_experts}
    assert {"adaptive", "kalman"} <= kinds


def test_predict_on_the_forecast_rows_returns_exactly_those_rows(fitted):
    model, _, future = fitted
    out = model.predict(future, date_col="ds")
    assert len(out) == len(future)
    assert out[FINAL].notna().all()


def test_predict_on_the_forecast_rows_equals_the_tail_of_predict_on_past_and_future(fitted):
    model, past, future = fitted
    alone = model.predict(future, date_col="ds")
    full = model.predict(pd.concat([past, future]), date_col="ds").tail(len(future))
    shared = [c for c in alone.columns if c in full.columns]
    assert FINAL in shared
    for column in shared:
        np.testing.assert_allclose(alone[column].to_numpy(), full[column].to_numpy(), rtol=0, atol=1e-9, err_msg=column)


def test_a_frame_inside_the_training_rows_gets_the_rows_before_it(fitted, monkeypatch):
    model, past, future = fitted
    seen = []
    original = model.data_manager.transform_test_data

    def spy(df_test, ctx):
        seen.append(len(df_test))
        return original(df_test, ctx)

    monkeypatch.setattr(model.data_manager, "transform_test_data", spy)
    frame = pd.concat([past.tail(40), future])
    out = model.predict(frame, date_col="ds")
    assert seen == [len(frame) + len(past) - 40]   # the training rows before the first date of the frame are put in front of it
    assert len(out) == len(frame)                  # and the output has exactly the rows given


def test_predict_online_on_the_forecast_rows_returns_exactly_those_rows(fitted):
    model, _, future = fitted
    out = model.predict_online(future, date_col="ds")
    assert len(out) == len(future)
    assert AutoTAM.FINAL_MODEL_COLUMN_ONLINE in out.columns
