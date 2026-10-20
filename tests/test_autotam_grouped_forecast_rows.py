# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Forecast rows of grouped data: the history of every group is put in front of the forecast rows of every group, so the forecast rows alone give
the forecasts of the same rows inside past + future; and the kept history has a known size.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from tam.model.autotam.auto_tam import AutoTAM

HORIZON = 5
EXPANSIONS = {"prior": True, "autofit": False, "kalman": True, "adaptive": True, "grid": False}


def _panel(n=90):
    rng = np.random.default_rng(21)
    parts = []
    for group, level in ((0, 20.0), (1, 60.0)):
        x1 = rng.normal(size=n)
        y = level + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + 2 * x1 + rng.normal(scale=0.3, size=n)
        parts.append(pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n, freq="D"), "g": group, "x1": x1, "y": y}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def grouped(tmp_path_factory):
    panel = _panel()
    past = panel[panel["ds"] < panel["ds"].max() - pd.Timedelta(days=HORIZON - 1)]
    future = panel[panel["ds"] >= panel["ds"].max() - pd.Timedelta(days=HORIZON - 1)].copy()
    future["y"] = np.nan
    model = AutoTAM("y ~ AutoPipe(x1)", lags=[HORIZON], n_experts=1, pop_size=4, use_opera=True, seed=3,
                    export_dir=str(tmp_path_factory.mktemp("grouped")))
    model.fit(past, date_col="ds", group_col="g", expansions=EXPANSIONS)
    return model, past, future


def _keyed(frame, forecast):
    out = forecast.copy()
    out.index = pd.MultiIndex.from_arrays([frame["g"].to_numpy(), frame["ds"].to_numpy()])
    return out


def test_the_history_of_every_group_is_kept(grouped):
    model, past, _ = grouped
    assert len(model.history_) == len(past) and set(model.history_["g"]) == {0, 1}


def test_the_forecast_rows_of_both_groups_alone_equal_their_rows_inside_past_plus_future(grouped):
    model, past, future = grouped
    alone = _keyed(future, model.predict(future, date_col="ds"))
    full_frame = pd.concat([past, future], ignore_index=True)
    full = _keyed(full_frame, model.predict(full_frame, date_col="ds")).loc[alone.index]
    assert len(alone) == 2 * HORIZON
    shared = [c for c in alone.columns if c in full.columns]
    assert shared
    for column in shared:
        np.testing.assert_allclose(alone[column].to_numpy(), full[column].to_numpy(), rtol=0, atol=1e-9, err_msg=column)


def test_the_size_of_the_kept_history_is_the_training_frame_and_a_pickle_of_the_model_stays_modest(grouped):
    model, past, _ = grouped
    frame_bytes = int(model.history_.memory_usage(deep=True).sum())
    pickled = len(pickle.dumps(model.history_))
    assert pickled < 40 * frame_bytes
    assert len(model.history_) == len(past)
