# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Apex membership of AutoTAM's two aggregation paths.

predict_online() aggregates the whole Apex pool (``apex_members_``); predict() averages only the
sparse weights kept at the end of validation (``weights_top10``). The expert pass and OperaTAM are
replaced by stubs, so the tests check the membership wiring without fitting a pipeline.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tam.model.autotam.auto_tam as auto_tam_module
from tam.model.autotam.auto_tam import AutoTAM


class _EqualWeightOpera:
    """OperaTAM stand-in: averages the experts named in its formula with equal weights."""

    formulas = []

    def __init__(self, formula, **kwargs):
        self.members = [term.strip()[2:-1] for term in formula.split("~")[1].split(" + ")]
        _EqualWeightOpera.formulas.append(formula)

    def predict_online(self, frame):
        out = pd.DataFrame({"prediction_opera": frame[self.members].mean(axis=1).to_numpy()})
        for member in self.members:
            out[f"weight_{member}"] = 1.0 / len(self.members)
        return out


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(auto_tam_module, "OperaTAM", _EqualWeightOpera)
    _EqualWeightOpera.formulas = []

    rng = np.random.default_rng(0)
    n = 20
    df_test = pd.DataFrame({"y": rng.normal(10.0, 1.0, n)})
    preds = pd.DataFrame({f"e{i}": rng.normal(10.0 + i, 1.0, n) for i in range(4)})

    auto = AutoTAM(formula="y ~ AutoPipe(x)", export_dir=str(tmp_path / "exports"))
    auto.ctx = SimpleNamespace(target="y", date_col=None, group_col=None)
    auto.trained_experts = [{"name": name} for name in preds.columns]
    auto.league_weights = {}
    auto.weights_top10 = {"e0": 0.6, "e1": 0.4}
    auto.apex_members_ = ["e0", "e1", "e2", "e3"]
    auto._expert_predictions = lambda df, date_col=None: (preds.copy(), df_test)
    return auto, df_test, preds


def test_predict_online_aggregates_the_whole_apex_pool(model):
    auto, df_test, preds = model
    out = auto.predict_online(df_test)
    assert _EqualWeightOpera.formulas[-1] == "y ~ l(e0) + l(e1) + l(e2) + l(e3)"
    np.testing.assert_allclose(out["AutoTAM_Apex_Online"], preds.mean(axis=1))


def test_predict_keeps_the_sparse_snapshot_weights(model):
    auto, df_test, preds = model
    out = auto.predict(df_test)
    np.testing.assert_allclose(out["AutoTAM_Apex_Ensemble"], 0.6 * preds["e0"] + 0.4 * preds["e1"])
    assert _EqualWeightOpera.formulas == []


def test_predict_online_falls_back_to_the_snapshot_without_a_stored_pool(model):
    auto, df_test, preds = model
    auto.apex_members_ = []
    auto.predict_online(df_test)
    assert _EqualWeightOpera.formulas[-1] == "y ~ l(e0) + l(e1)"
