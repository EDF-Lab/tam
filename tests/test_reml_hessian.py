# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
The analytic Hessian of the REML criterion in ``log lambda`` against central differences of the analytic gradient.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.common.utils import _balance_groups, _ensure_dummies
from tam.model._math import _compute_weighted_covariances
from tam.model._reml import PenaltyBasis, reml_objective


def _system(formula, frame, groups=1):
    model = ta.StaticTAM(formula)
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    x, y, _ = model._prepare_data(balanced, target_col=model.target_col_)
    phi = model._build_design_matrix(x).double()
    cov_x, cov_xy = _compute_weighted_covariances(phi, y.double(), torch.eye(1, dtype=torch.float64))
    y_sq = (y.double() ** 2).sum(dim=(1, 2))
    basis = PenaltyBasis(model.effects_list_)
    return model, basis, cov_x, cov_xy, y_sq, x.shape[1]


def _system_x(formula, frame):
    model = ta.StaticTAM(formula)
    dummied = _ensure_dummies(frame.copy(), model.group_col_, model.date_col_)
    _, balanced = _balance_groups(dummied, model.group_col_, model.date_col_, method="drop")
    return model._prepare_data(balanced, target_col=model.target_col_)[0]


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    return pd.DataFrame({"x": x, "z": z, "y": np.sin(2 * np.pi * x) + 0.5 * z ** 2 + 0.3 * rng.normal(size=n)})


def _reference(basis, cov_x, cov_xy, y_sq, n, known_scale, rho):
    """Value, gradient and Hessian of minus the criterion by automatic differentiation of a dense, independent evaluation."""
    size = cov_x.shape[-1]
    reduced_basis = basis.basis.double()

    def criterion(r):
        penalty = basis.constant.clone()
        for c, (begin, end, q) in enumerate(basis.blocks):
            padded = torch.zeros(size, size, dtype=torch.float64)
            padded[begin:end, begin:end] = q
            penalty = penalty + torch.exp(r[c]) * padded
        log_det_s = torch.linalg.slogdet(reduced_basis.T @ (n * penalty) @ reduced_basis)[1]
        dof = n - basis.null_dim
        total = 0.0
        for g in range(cov_x.shape[0]):
            system = cov_x[g] + n * penalty + 1e-6 * n * torch.eye(size, dtype=torch.float64)
            beta = torch.linalg.solve(system, cov_xy[g])
            deviance = y_sq[g] - (beta * cov_xy[g]).sum()
            log_det_a = torch.linalg.slogdet(system)[1]
            if known_scale is None:
                total = total - 0.5 * log_det_a + 0.5 * log_det_s - 0.5 * dof * torch.log(deviance / dof)
            else:
                total = total - 0.5 * log_det_a + 0.5 * log_det_s - deviance / (2.0 * known_scale)
        return -total

    point = torch.tensor(rho, dtype=torch.float64)
    return (
        float(criterion(point)),
        torch.autograd.functional.jacobian(criterion, point).numpy(),
        torch.autograd.functional.hessian(criterion, point).numpy(),
    )


def _check(formula, frame, rho, known_scale=None, groups=1):
    model, basis, cov_x, cov_xy, y_sq, n = _system(formula, frame)
    if groups > 1:  # the same design and weights, a different response per group
        rng = np.random.default_rng(11)
        phi = model._build_design_matrix(_system_x(formula, frame)).double()[0]
        cov_x = cov_x.expand(groups, -1, -1).contiguous()
        ys = [torch.as_tensor(frame["y"].to_numpy() * (1.0 + 0.4 * g) + 0.2 * g * rng.normal(size=len(frame)), dtype=torch.float64) for g in range(groups)]
        cov_xy = torch.stack([phi.T @ y.view(-1, 1) for y in ys])
        y_sq = torch.stack([(y ** 2).sum() for y in ys])
    rho = np.asarray(rho[: basis.n_coordinates], dtype=float)
    m = basis.n_coordinates
    hess = np.zeros((m, m))
    value, grad = reml_objective(rho, cov_x, cov_xy, y_sq, basis, n, known_scale, compute_hess=True, hess_out=hess)
    value0, grad0 = reml_objective(rho, cov_x, cov_xy, y_sq, basis, n, known_scale)
    assert value == value0 and np.array_equal(grad, grad0)
    assert np.allclose(hess, hess.T, atol=1e-12)
    ref_value, ref_grad, ref_hess = _reference(basis, cov_x, cov_xy, y_sq, n, known_scale, rho)
    assert abs(value - ref_value) < 1e-8 * max(1.0, abs(ref_value))
    assert np.allclose(grad, ref_grad, rtol=1e-6, atol=1e-7 * np.abs(ref_grad).max())
    assert np.allclose(hess, ref_hess, rtol=1e-6, atol=1e-7 * np.abs(ref_hess).max())  # the null cut of the penalty subspace (1e-9) limits the agreement of the dense group to about 1e-8, np.abs(hess - ref_hess).max()

    def gradient_at(point):
        return reml_objective(point, cov_x, cov_xy, y_sq, basis, n, known_scale)[1]

    def central(h):
        out = np.zeros((m, m))
        for j in range(m):
            step = np.zeros(m)
            step[j] = h
            out[:, j] = (gradient_at(rho + step) - gradient_at(rho - step)) / (2.0 * h)
        return out

    h = 3e-2  # Richardson: the truncation error is O(h^4); a smaller step lets the rounding noise of the gradient (about 1e-13 / h) exceed the tolerance
    numeric = (4.0 * central(h / 2.0) - central(h)) / 3.0
    numeric = 0.5 * (numeric + numeric.T)
    assert np.allclose(hess, numeric, rtol=1e-4, atol=1e-6 * np.abs(hess).max()), np.abs(hess - numeric).max()
    return hess


@pytest.mark.parametrize("known_scale", [None, 0.5])
def test_hessian_scalar_terms(known_scale):
    _check("y ~ s(x, k=12) + s(z, k=8)", _frame(), [0.3, 1.1, -0.4], known_scale)


@pytest.mark.parametrize("known_scale", [None, 0.5])
def test_hessian_commuting_tensor_product(known_scale):
    hess = _check("y ~ te(s(x, k=6), s(z, k=6))", _frame(), [0.2, 0.9, 0.5], known_scale)
    assert abs(hess[0, 1]) > 0.0


@pytest.mark.parametrize("known_scale", [None, 0.5])
def test_hessian_with_a_fixed_penalty_and_several_effects(known_scale):
    _check("y ~ s(x, k=10, m_pen=3) + te(s(x, k=5), s(z, k=5)) + s(z, k=6)", _frame(), [0.1, -0.3, 0.6, 1.2, 0.4, 0.2, -0.5], known_scale)


@pytest.mark.parametrize("known_scale", [None, 0.5])
def test_hessian_with_three_groups(known_scale):
    _check("y ~ s(x, k=10) + te(s(x, k=5), s(z, k=5)) + s(z, k=6)", _frame(), [0.1, -0.3, 0.6, 1.2, 0.4], known_scale, groups=3)


@pytest.mark.parametrize("known_scale", [None, 0.5])
def test_hessian_with_a_larger_design(known_scale):
    _check("y ~ s(x, k=60) + s(z, k=60)", _frame(n=500), [0.3, 1.1, -0.4], known_scale)
