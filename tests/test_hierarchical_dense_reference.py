# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Characterisation tests for ``HierarchicalTAM``: what the joint solver computes today.

The three solve paths (direct, dense CG, matrix-free CG) must agree, ``lambda_p_hier = 0`` must be the independent ``StaticTAM`` fits,
and the assembled global system must be the one of the mathematical definition
(diagonal self-terms plus ``lambda ΦpᵀΦp`` for the parent, ``-lambda ΦpᵀΦc`` parent-child, ``+lambda ΦciᵀΦcj`` between siblings).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tam.common.utils import _ensure_dummies
from tam.model import hierarchical as hier_module
from tam.model.additive import StaticTAM
from tam.model.hierarchical import HierarchicalTAM

FORMULA = "load ~ l(temperature) + s(temperature, k=5)"
STRUCTURE = {"Total": ["RegionA", "RegionB"]}


def _data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n)
    frames = []
    for node, base, slope in [("Total", 100.0, 1.5), ("RegionA", 60.0, 1.0), ("RegionB", 40.0, 0.5)]:
        temp = rng.uniform(-5, 30, n)
        frames.append(pd.DataFrame({
            "timestamp": dates,
            "node": node,
            "temperature": temp,
            "load": base + slope * temp + 0.05 * temp ** 2 + rng.normal(0, 2.0, n),
        }))
    return pd.concat(frames, ignore_index=True)


def _model(lam=1.0, formula=FORMULA):
    return HierarchicalTAM(structure=STRUCTURE, formulas=formula, node_col="node", date_col="timestamp", lambda_p_hier=lam)


def _coefficients(model):
    return {node: model.sub_models[node].coefficients_.detach().cpu().numpy().ravel() for node in model.nodes}


def test_lambda_zero_is_three_independent_static_fits():
    df = _data()
    joint = _model(lam=0.0).fit(df)
    for node, coefficients in _coefficients(joint).items():
        alone = StaticTAM(formula=FORMULA, date_col="timestamp").fit(df[df["node"] == node].reset_index(drop=True))
        np.testing.assert_allclose(coefficients, alone.coefficients_.detach().cpu().numpy().ravel(), rtol=1e-8, atol=1e-10)


def test_predictions_of_the_joint_fit_are_pinned():
    """Black-box pin of today's numbers (lambda_p_hier = 1): a refactor of ``fit`` must reproduce them."""
    model = _model(lam=1.0).fit(_data())
    preds = model.predict(_data())
    summary = preds.groupby("node")["Estimatedload"].agg(["mean", "std"])
    first = preds[preds["node"] == "Total"]["Estimatedload"].to_numpy()[:3]
    np.testing.assert_allclose(summary.loc["Total", "mean"], PINNED["Total_mean"], rtol=1e-8)
    np.testing.assert_allclose(summary.loc["RegionA", "mean"], PINNED["RegionA_mean"], rtol=1e-8)
    np.testing.assert_allclose(summary.loc["RegionB", "mean"], PINNED["RegionB_mean"], rtol=1e-8)
    np.testing.assert_allclose(summary.loc["Total", "std"], PINNED["Total_std"], rtol=1e-8)
    np.testing.assert_allclose(first, PINNED["Total_first"], rtol=1e-8)


PINNED = {
    "Total_mean": 136.4747819939654,
    "RegionA_mean": 84.67704768487769,
    "RegionB_mean": 55.10748005641436,
    "Total_std": 18.029778579360247,
    "Total_first": [139.45128161872583, 118.5324962274291, 118.55007888058387],
}


def _force_path(monkeypatch, path):
    """Route the solve through the direct, dense-CG or matrix-free CG path by mocking the memory probe."""
    direct, available = {"direct": (True, 1e12), "dense_cg": (False, 1e12), "matrix_free": (False, 0.0)}[path]
    monkeypatch.setattr(hier_module, "can_fit_dense_matrix", lambda *a, **k: direct)
    monkeypatch.setattr(hier_module.hw, "get_available_memory", lambda: available)


def _fit(monkeypatch, path, df):
    _force_path(monkeypatch, path)
    model = _model(lam=1.0).fit(df)
    return _coefficients(model), model.predict(df)["Estimatedload"].to_numpy()


def test_the_three_solve_paths_agree(monkeypatch):
    """CG stops at ``tol=1e-6`` on an ill-conditioned system: coefficients agree to about 1e-3 of the largest one, predictions to 1e-5 relative.
    The two CG paths solve the same system, so they agree much more closely with each other."""
    df = _data()
    direct, direct_pred = _fit(monkeypatch, "direct", df)
    dense, dense_pred = _fit(monkeypatch, "dense_cg", df)
    free, free_pred = _fit(monkeypatch, "matrix_free", df)
    scale = max(np.abs(c).max() for c in direct.values())
    for node in direct:
        np.testing.assert_allclose(dense[node], direct[node], rtol=0, atol=3e-3 * scale)
        np.testing.assert_allclose(free[node], dense[node], rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(dense_pred, direct_pred, rtol=1e-5)
    np.testing.assert_allclose(free_pred, direct_pred, rtol=1e-5)


def _textbook_system(model, df, lam):
    """The global matrix from the definition, built from each node's own design matrix (G = 1)."""
    df = _ensure_dummies(df, model.group_col, model.date_col)
    phi, y, penalty, index = {}, {}, {}, {}
    offset = 0
    for node in model.nodes:
        sub = model.sub_models[node]
        x, target, _ = sub._prepare_data(df[df["node"] == node].reset_index(drop=True), target_col=sub.target_col_)
        phi[node] = sub._build_design_matrix(x).squeeze(0)
        y[node] = target.squeeze(0)
        penalty[node] = sub._build_penalty_matrix()
        if penalty[node].dim() == 3:
            penalty[node] = penalty[node].squeeze(0)
        index[node] = (offset, offset + phi[node].shape[1])
        offset += phi[node].shape[1]
    cov_x = torch.zeros((offset, offset), dtype=phi["Total"].dtype)
    cov_xy = torch.zeros((offset, 1), dtype=phi["Total"].dtype)
    pen = torch.zeros((offset, offset), dtype=phi["Total"].dtype)
    for node in model.nodes:
        a, b = index[node]
        cov_x[a:b, a:b] += phi[node].mT @ phi[node]
        cov_xy[a:b] += phi[node].mT @ y[node]
        pen[a:b, a:b] = penalty[node]
    for parent, children in STRUCTURE.items():
        p0, p1 = index[parent]
        cov_x[p0:p1, p0:p1] += lam * phi[parent].mT @ phi[parent]
        for child in children:
            c0, c1 = index[child]
            cov_x[c0:c1, c0:c1] += lam * phi[child].mT @ phi[child]
            cov_x[p0:p1, c0:c1] += -lam * phi[parent].mT @ phi[child]
            cov_x[c0:c1, p0:p1] += -lam * phi[child].mT @ phi[parent]
            for sibling in children:
                if sibling != child:
                    s0, s1 = index[sibling]
                    cov_x[c0:c1, s0:s1] += lam * phi[child].mT @ phi[sibling]
    return cov_x, cov_xy, pen, index


@pytest.mark.parametrize("lam", [0.0, 1.0, 7.5])
def test_the_assembled_system_is_the_definition(lam, monkeypatch):
    df = _data(n=40)
    _force_path(monkeypatch, "direct")
    model = _model(lam=lam)
    system = model._assemble_system(df)
    cov_x, cov_xy, pen, index = _textbook_system(model, df, lam)
    assert system.node_to_idx == index
    assert system.n_samples == 40
    np.testing.assert_allclose(system.cov_X.cpu().numpy(), cov_x.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(system.cov_XY.cpu().numpy(), cov_xy.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(system.P.cpu().numpy(), pen.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(system.cov_X.cpu().numpy(), system.cov_X.cpu().numpy().T, atol=1e-10)


def test_solving_the_assembled_system_is_fit(monkeypatch):
    df = _data(n=40)
    _force_path(monkeypatch, "direct")
    model = _model(lam=1.0)
    beta = model._solve(model._assemble_system(df))
    fitted = _coefficients(_model(lam=1.0).fit(df))
    for node, (a, b) in model._assemble_system(df).node_to_idx.items():
        np.testing.assert_allclose(beta[a:b].cpu().numpy().ravel(), fitted[node], rtol=1e-10, atol=1e-12)
