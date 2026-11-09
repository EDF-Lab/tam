# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``HierarchicalTAM`` with a ``group_col``: one hierarchical system per group (for instance per half-hour of the day).

The group axis stays in the tensors ``(G, T, F)`` of every node, the global system is ``(G, D, D)`` and solved in one batched call. The
groups do not talk to each other: the joint fit on G groups equals G separate fits, one per group.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.model import hierarchical as hier_module
from tam.model.hierarchical import HierarchicalTAM

FORMULA = "load ~ l(temperature) + s(temperature, k=5)"
STRUCTURE = {"Total": ["RegionA", "RegionB"]}
SLOTS = [0, 1, 2]


def _data(slots=SLOTS, n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n)
    frames = []
    for slot in slots:
        temp = rng.uniform(-5, 30, n)
        loads = {
            "RegionA": 60.0 + 5 * slot + (1.0 + 0.2 * slot) * temp + rng.normal(0, 2, n),
            "RegionB": 40.0 - 3 * slot + 0.5 * temp + rng.normal(0, 2, n),
        }
        loads["Total"] = loads["RegionA"] + loads["RegionB"] + rng.normal(0, 1, n)
        for node in ("Total", "RegionA", "RegionB"):
            frames.append(pd.DataFrame({"timestamp": dates, "slot": slot, "node": node, "temperature": temp, "load": loads[node]}))
    return pd.concat(frames, ignore_index=True)


def _model(lam=1.0, group_col="slot"):
    return HierarchicalTAM(structure=STRUCTURE, formulas=FORMULA, node_col="node", group_col=group_col, date_col="timestamp",
                           lambda_p_hier=lam)


def _force_path(monkeypatch, path):
    direct, available = {"direct": (True, 1e12), "dense_cg": (False, 1e12), "matrix_free": (False, 0.0)}[path]
    monkeypatch.setattr(hier_module, "can_fit_dense_matrix", lambda *a, **k: direct)
    monkeypatch.setattr(hier_module.hw, "get_available_memory", lambda: available)


def _coefficients(model, node):
    return model.sub_models[node].coefficients_.detach().cpu().numpy()


def test_every_node_holds_one_coefficient_vector_per_group():
    model = _model().fit(_data())
    for node in model.nodes:
        assert _coefficients(model, node).shape[0] == len(SLOTS)


def test_the_joint_fit_on_three_groups_is_three_separate_fits():
    df = _data()
    joint = _model().fit(df)
    for g, slot in enumerate(SLOTS):
        alone = _model().fit(df[df["slot"] == slot].reset_index(drop=True))
        for node in joint.nodes:
            np.testing.assert_allclose(_coefficients(joint, node)[g], _coefficients(alone, node)[0], rtol=1e-8, atol=1e-10)


def test_group_with_lambda_zero_is_independent_static_fits():
    from tam.model.additive import StaticTAM

    df = _data()
    joint = _model(lam=0.0).fit(df)
    for node in joint.nodes:
        alone = StaticTAM(formula=FORMULA, group_col="slot", date_col="timestamp").fit(df[df["node"] == node].reset_index(drop=True))
        np.testing.assert_allclose(_coefficients(joint, node), alone.coefficients_.detach().cpu().numpy(), rtol=1e-8, atol=1e-10)


def test_predict_covers_every_group_and_node():
    df = _data()
    preds = _model().fit(df).predict(df)
    assert len(preds) == len(df)
    assert set(preds["slot"]) == set(SLOTS) and not preds["Estimatedload"].isna().any()


def test_a_node_missing_a_group_is_refused_by_the_alignment():
    df = _data()
    df = df[~((df["node"] == "RegionB") & (df["slot"] == 2))].reset_index(drop=True)
    with pytest.raises(ValueError, match="share the same"):
        _model().fit(df)


def test_the_nodes_must_hold_the_same_groups():
    """Without a date column the nodes are matched by position: equal row counts with different group labels are still refused."""
    df = _data().drop(columns="timestamp")
    df.loc[(df["node"] == "RegionB") & (df["slot"] == 2), "slot"] = 3
    model = HierarchicalTAM(structure=STRUCTURE, formulas=FORMULA, node_col="node", group_col="slot")
    with pytest.raises(ValueError, match="same groups"):
        model.fit(df)


def test_the_three_solve_paths_agree_on_three_groups(monkeypatch):
    df = _data()
    fits = {}
    for path in ("direct", "dense_cg", "matrix_free"):
        _force_path(monkeypatch, path)
        model = _model().fit(df)
        fits[path] = {node: _coefficients(model, node) for node in model.nodes}
    scale = max(np.abs(c).max() for c in fits["direct"].values())
    for node in fits["direct"]:
        np.testing.assert_allclose(fits["dense_cg"][node], fits["direct"][node], rtol=0, atol=3e-3 * scale)
        np.testing.assert_allclose(fits["matrix_free"][node], fits["dense_cg"][node], rtol=0, atol=1e-6 * scale)


def test_the_dense_system_has_one_matrix_per_group(monkeypatch):
    _force_path(monkeypatch, "direct")
    system = _model()._assemble_system(_data())
    d = system.total_coeffs
    assert system.n_groups == len(SLOTS)
    assert tuple(system.cov_X.shape) == (len(SLOTS), d, d) and tuple(system.cov_XY.shape) == (len(SLOTS), d, 1)
    assert tuple(system.P.shape) == (d, d)


def test_the_memory_oracle_counts_the_groups(monkeypatch):
    """Dense covariance for G groups needs G·D²·8 bytes against 40 % of the available memory: three groups go matrix-free where one group stays dense."""
    df = _data()
    _force_path(monkeypatch, "direct")
    d = _model()._assemble_system(df[df["slot"] == 0].reset_index(drop=True)).total_coeffs
    monkeypatch.setattr(hier_module, "can_fit_dense_matrix", lambda *a, **k: False)
    monkeypatch.setattr(hier_module.hw, "get_available_memory", lambda: 5.0 * d * d * 8)
    assert _model()._assemble_system(df[df["slot"] == 0].reset_index(drop=True)).cov_X is not None
    assert _model()._assemble_system(df).cov_X is None
