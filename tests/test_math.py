r"""
Unit tests for the core mathematical solver functions in ``tam.model._math``.

These cover the pure linear-algebra building blocks of the primal solver:
covariance assembly, the regularized normal-equation solve, prediction,
additive decomposition, the GCV score, and the matrix-free CG solver.
"""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model._math import (
    _compute_weighted_covariances,
    solve_linear_system,
    _predict_from_coeffs,
    _decompose_prediction_tensor,
    compute_gcv_score,
    solve_sparse_cg,
)
from tam.model.spectrum import OffsetEffect, LinearEffect, build_phi_from_effects


def _make_system(n_groups=2, n_samples=20, n_coeffs=4, seed=0):
    """Builds a well-posed batched regression system y = phi @ true_beta."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    phi = torch.rand(n_groups, n_samples, n_coeffs, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    true_beta = torch.rand(n_groups, n_coeffs, 1, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    y = phi @ true_beta
    return phi, y, true_beta


def test_compute_weighted_covariances_shapes_and_symmetry():
    phi, y, _ = _make_system()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    cov_X, cov_XY = _compute_weighted_covariances(phi, y, loss)

    assert cov_X.shape == (2, 4, 4)
    assert cov_XY.shape == (2, 4, 1)
    # Phi^T Phi is symmetric.
    assert torch.allclose(cov_X, cov_X.mT, atol=1e-8)


def test_solve_linear_system_recovers_coefficients():
    phi, y, true_beta = _make_system()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    cov_X, cov_XY = _compute_weighted_covariances(phi, y, loss)

    # Zero penalty -> ordinary least squares; the exact solution is recoverable.
    penalty = torch.zeros(4, 4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    beta = solve_linear_system(cov_X, cov_XY, penalty, n_samples=1)

    assert beta.shape == (2, 4, 1)
    assert torch.allclose(beta, true_beta, atol=1e-3)


def test_solve_linear_system_penalty_shrinks_coefficients():
    phi, y, _ = _make_system()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    cov_X, cov_XY = _compute_weighted_covariances(phi, y, loss)

    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    beta_light = solve_linear_system(cov_X, cov_XY, penalty, n_samples=1)
    beta_heavy = solve_linear_system(cov_X, cov_XY, penalty, n_samples=1000)

    # Stronger regularization must pull coefficient norms toward zero.
    assert beta_heavy.norm() < beta_light.norm()


def test_solve_linear_system_accepts_sparse_penalty():
    phi, y, _ = _make_system()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    cov_X, cov_XY = _compute_weighted_covariances(phi, y, loss)

    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype()).to_sparse()
    beta = solve_linear_system(cov_X, cov_XY, penalty, n_samples=10)
    assert beta.shape == (2, 4, 1)
    assert torch.isfinite(beta).all()


def test_predict_from_coeffs_matches_manual_product():
    phi, _, true_beta = _make_system()
    preds = _predict_from_coeffs(phi, true_beta)
    assert preds.shape == (2, 20, 1)
    assert torch.allclose(preds, phi @ true_beta)


def test_decompose_prediction_tensor_sums_to_total():
    effects = [
        OffsetEffect(lambda_p=1e-3, extrapolate="continue"),
        LinearEffect("x", scaled=1.0, lambda_p=1.0, extrapolate="continue"),
    ]
    g = torch.Generator(device="cpu").manual_seed(3)
    x = (torch.rand(2, 15, 1, generator=g, dtype=torch.get_default_dtype()) * 2 - 1).to(TORCH_DEVICE)
    phi = build_phi_from_effects(x, effects, feature_columns=["x"])
    beta = torch.rand(2, phi.shape[-1], 1, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)

    parts = _decompose_prediction_tensor(phi, beta, effects)

    assert set(parts.keys()) == {"offset", "x"}
    summed = sum(parts.values())
    total = (phi @ beta).squeeze(-1)
    assert torch.allclose(summed, total, atol=1e-8)


def test_decompose_prediction_tensor_prefixes_name_collisions():
    """Two effects on the same feature must be disambiguated by a type prefix."""
    effects = [
        LinearEffect("x", scaled=1.0, lambda_p=1.0, extrapolate="continue"),
        LinearEffect("x", scaled=2.0, lambda_p=1.0, extrapolate="continue"),
    ]
    g = torch.Generator(device="cpu").manual_seed(4)
    x = (torch.rand(1, 10, 1, generator=g, dtype=torch.get_default_dtype()) * 2 - 1).to(TORCH_DEVICE)
    phi = build_phi_from_effects(x, effects, feature_columns=["x"])
    beta = torch.rand(1, phi.shape[-1], 1, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)

    parts = _decompose_prediction_tensor(phi, beta, effects)
    # Collision on 'x' -> linear prefix 'l_x'
    assert "l_x" in parts


def test_compute_gcv_score_returns_finite_scalar():
    phi, y, _ = _make_system(n_groups=2, n_samples=30, n_coeffs=4, seed=5)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    cov_X, cov_XY = _compute_weighted_covariances(phi, y, loss)
    Y_sq = torch.sum(y ** 2, dim=1).squeeze(-1)  # shape (B,)
    penalty = torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    score = compute_gcv_score(cov_X, cov_XY, Y_sq, penalty, lambda_p=1.0, n_samples=30, gamma=1.4)
    assert score.numel() == 1
    assert torch.isfinite(score)
    assert score.item() >= 0.0


def test_solve_sparse_cg_matches_direct_solve():
    """Matrix-free CG must converge to the same solution as a direct solve."""
    g = torch.Generator(device="cpu").manual_seed(6)
    M = torch.rand(5, 5, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    A = M @ M.T + 5 * torch.eye(5, device=TORCH_DEVICE, dtype=torch.get_default_dtype())  # SPD
    b = torch.rand(5, 1, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)

    x_cg = solve_sparse_cg(lambda v: A @ v, b, tol=1e-10, max_iter=200)
    x_direct = torch.linalg.solve(A, b)
    assert torch.allclose(x_cg, x_direct, atol=1e-5)


def test_solve_sparse_cg_batched():
    g = torch.Generator(device="cpu").manual_seed(7)
    batch = 3
    M = torch.rand(batch, 4, 4, generator=g, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)
    A = M @ M.mT + 5 * torch.eye(4, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    b = torch.rand(batch, 4, 1, dtype=torch.get_default_dtype()).to(TORCH_DEVICE)

    x_cg = solve_sparse_cg(lambda v: A @ v, b, tol=1e-10, max_iter=200)
    x_direct = torch.linalg.solve(A, b)
    assert x_cg.shape == (batch, 4, 1)
    assert torch.allclose(x_cg, x_direct, atol=1e-5)
