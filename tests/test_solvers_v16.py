# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import torch
from tam.model._math import solve_linear_system, solve_sparse_cg, solve_block_qr, compute_gcv_score, compute_gcv_score_and_grad, solve_sqrt_group_lasso, compute_scale_stability_penalty

def test_next_gen_solvers_and_pivotal_sparsity():
    torch.manual_seed(11)
    N, K = 160, 12
    Phi = torch.randn((1, N, K), dtype=torch.float64)
    true_beta = torch.zeros((1, K, 1), dtype=torch.float64)
    true_beta[:, :4, 0] = torch.tensor([1.5, -1.0, 0.8, -0.5], dtype=torch.float64)
    Y = Phi @ true_beta + 0.02 * torch.randn((1, N, 1), dtype=torch.float64)
    P = 0.05 * torch.eye(K, dtype=torch.float64)

    cov_X = Phi.mT @ Phi
    cov_XY = Phi.mT @ Y
    beta_direct = solve_linear_system(cov_X, cov_XY, P, N)

    # 1. Incremental Block-QR matches solve_linear_system to < 1e-10
    beta_qr = solve_block_qr(
        [Phi[:, :60, :], Phi[:, 60:110, :], Phi[:, 110:, :]],
        [Y[:, :60, :], Y[:, 60:110, :], Y[:, 110:, :]],
        P,
        n_samples=N,
    )
    assert torch.max(torch.abs(beta_direct - beta_qr)).item() < 1e-9

    # 2. Preconditioned CG with Jacobi preconditioner matches direct solve
    A_mat = cov_X[0] + N * P + (1e-6 * N) * torch.eye(K, dtype=torch.float64)
    jacobi_inv = 1.0 / torch.diag(A_mat).unsqueeze(-1)
    beta_pcg = solve_sparse_cg(
        lambda v: A_mat @ v,
        cov_XY[0],
        tol=1e-10,
        max_iter=200,
        preconditioner=jacobi_inv,
    )
    assert torch.max(torch.abs(beta_direct[0] - beta_pcg)).item() < 1e-7

    # 3. Hutchinson GCV score and Wood analytical GCV gradient
    Y_sq = torch.sum(Y ** 2, dim=1).squeeze(-1)
    gcv_exact = compute_gcv_score(cov_X, cov_XY, Y_sq, P, 1.0, N, gamma=1.4, trace_method="exact")
    gcv_hutch = compute_gcv_score(cov_X, cov_XY, Y_sq, P, 1.0, N, gamma=1.4, trace_method="hutchinson", n_probes=128)
    assert abs(float(gcv_exact.item()) - float(gcv_hutch.item())) / float(gcv_exact.item()) < 0.05

    S1 = torch.eye(6, dtype=torch.float64)
    S2 = torch.eye(6, dtype=torch.float64)
    alphas = torch.tensor([-1.5, -1.0], dtype=torch.float64)
    val0, grad_a = compute_gcv_score_and_grad(cov_X, cov_XY, Y_sq, [S1, S2], [(0, 6), (6, 12)], alphas, N, gamma=1.4)
    eps = 1e-6
    for j in range(2):
        a_plus = alphas.clone()
        a_minus = alphas.clone()
        a_plus[j] += eps
        a_minus[j] -= eps
        v_p, _ = compute_gcv_score_and_grad(cov_X, cov_XY, Y_sq, [S1, S2], [(0, 6), (6, 12)], a_plus, N, gamma=1.4)
        v_m, _ = compute_gcv_score_and_grad(cov_X, cov_XY, Y_sq, [S1, S2], [(0, 6), (6, 12)], a_minus, N, gamma=1.4)
        fd_grad = (v_p.item() - v_m.item()) / (2.0 * eps)
        assert abs(grad_a[j].item() - fd_grad) < 1e-5

    # 4. Square-Root Group LASSO active group recovery & scale-stability penalty
    spans = [(0, 4), (4, 8), (8, 12)]
    theta_gl, active_groups = solve_sqrt_group_lasso(Phi[0], Y[0], spans, c_scale=1.1, penalty_matrix=P)
    assert 0 in active_groups
    assert theta_gl.shape == (K, 1)
    pen_var = compute_scale_stability_penalty(Phi[0, :, 0])
    assert float(pen_var.item()) >= 0.0


def test_sqrt_group_lasso_active_groups_do_not_depend_on_the_scale_of_y():
    """The active groups are read on the standardized solution, so rescaling the response does not change them."""
    torch.manual_seed(3)
    T, spans = 400, [(0, 4), (4, 8), (8, 12), (12, 16)]
    X = torch.randn(T, 16, dtype=torch.float64)
    theta_true = torch.zeros(16, 1, dtype=torch.float64)
    theta_true[0:4] = 2.0
    theta_true[8:12] = -1.5
    y = X @ theta_true + 0.05 * torch.randn(T, 1, dtype=torch.float64)
    P = torch.eye(16, dtype=torch.float64)
    _, active_ref = solve_sqrt_group_lasso(X, y, spans, c_scale=1.1, penalty_matrix=P)
    assert active_ref == [0, 2]
    for scale in (1e-4, 1e-6, 1e4):
        _, active = solve_sqrt_group_lasso(X, y * scale, spans, c_scale=1.1, penalty_matrix=P)
        assert active == active_ref, (scale, active)
