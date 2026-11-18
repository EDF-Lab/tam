# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""StaticTAM with a stateful effect: the end-of-training state of every group, and continuing a prediction from it."""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model._base import BaseTAM
from tam.model.additive import StaticTAM
from tam.model.spectrum import SingleIndexEffect

N_TRAIN, N_NEW = 150, 60


def _panel(n_groups: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = np.cumsum(rng.normal(size=N_TRAIN + 2 * N_NEW)) * 0.1
        y = np.tanh(x) + 0.3 * np.tanh(np.roll(x, 3)) + 0.05 * rng.normal(size=x.size)
        rows.append(pd.DataFrame({"g": g, "t": np.arange(x.size), "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


FORMULA = "y ~ si(x, lag_window=6, k=6)"


def _split(df):
    train = df[df.t < N_TRAIN]
    val = df[(df.t >= N_TRAIN) & (df.t < N_TRAIN + N_NEW)]
    test = df[df.t >= N_TRAIN + N_NEW]
    return train, val, test


def _si_index(model) -> int:
    return [i for i, e in enumerate(model.effects_list_) if isinstance(e, SingleIndexEffect)][0]


def _fit(df, n_groups=3):
    train, val, test = _split(df)
    return StaticTAM(FORMULA, group_col="g", date_col="t").fit(train), train, val, test


def test_predict_is_idempotent_and_stateless_by_default():
    m, _, val, test = _fit(_panel(3))
    p_val_1 = m.predict(val)["Estimatedy"].to_numpy()
    m.predict(test)
    p_val_2 = m.predict(val)["Estimatedy"].to_numpy()
    assert np.array_equal(p_val_1, p_val_2)
    fresh, _, _, _ = _fit(_panel(3))
    assert np.array_equal(m.predict(test, continue_state=False)["Estimatedy"].to_numpy(), fresh.predict(test)["Estimatedy"].to_numpy())


def test_continue_state_equals_the_prediction_of_the_concatenated_frame():
    df = _panel(3)
    m, train, val, _ = _fit(df)
    carried = m.predict(val, continue_state=True)["Estimatedy"].to_numpy()
    whole = m.predict(pd.concat([train, val], ignore_index=True))
    expected = whole[whole.t >= N_TRAIN]["Estimatedy"].to_numpy()
    assert np.allclose(carried, expected, atol=1e-8)
    assert not np.allclose(carried, m.predict(val)["Estimatedy"].to_numpy(), atol=1e-8)   # the start of the stateless pass differs


def test_state_has_one_row_per_group_whatever_the_chunking(monkeypatch):
    df = _panel(5)
    train, val, _test = _split(df)
    monkeypatch.setattr(BaseTAM, "_group_chunks", lambda self, x: [slice(s, min(s + 2, x.shape[0])) for s in range(0, x.shape[0], 2)])
    m = StaticTAM(FORMULA, group_col="g", date_col="t").fit(train)
    state = m._effect_states_[_si_index(m)]["hist"]
    assert state.shape[0] == 5
    monkeypatch.undo()
    ref = StaticTAM(FORMULA, group_col="g", date_col="t").fit(train)
    assert np.allclose(state.numpy(), ref._effect_states_[_si_index(ref)]["hist"].numpy())


def test_continue_state_with_chunked_prediction(monkeypatch):
    from tam.model import _base

    df = _panel(5)
    m, train, val, _ = _fit(df, 5)
    expected = m.predict(val, continue_state=True)["Estimatedy"].to_numpy()
    d = m._build_design_matrix(torch.zeros(1, 1, len(m.features_config_["features"]), dtype=torch.float64)).shape[-1]
    per_group = N_NEW * d * 8 * 3.0
    monkeypatch.setattr(_base.hw, "get_available_memory", lambda: per_group * 2.5 / 0.8)   # two groups at a time
    chunked = m.predict(val, continue_state=True)["Estimatedy"].to_numpy()
    assert np.allclose(chunked, expected, atol=1e-10)


def test_auto_fit_and_grid_search_fit_fill_the_states():
    df = _panel(3)
    train, val, _ = _split(df)
    auto = StaticTAM(FORMULA, group_col="g", date_col="t")
    auto.auto_fit(train, alpha_p_bounds=(-4.0, 1.0), number_of_steps=3)
    assert auto._effect_states_[_si_index(auto)]["hist"].shape == (3, 5, 1)
    best = StaticTAM("y ~ si(x, lag_window=6, k=grid_k)", group_col="g", date_col="t").grid_search_fit(train, val, {"grid_k": [4, 6]})
    assert best._effect_states_[_si_index(best)]["hist"].shape == (3, 5, 1)
    p = best.predict(val, continue_state=True)
    assert np.isfinite(p["Estimatedy"].to_numpy()).all()


def test_continue_state_needs_the_same_groups():
    df = _panel(3)
    m, _, val, _ = _fit(df)
    other = val[val.g != 2]
    with pytest.raises(ValueError, match="same groups"):
        m.predict(other, continue_state=True)
