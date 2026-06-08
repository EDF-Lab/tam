r"""
Unit tests for the solver dispatchers and global assembly helpers:
``tam.model._dispatcher`` (routing, evaluation, decomposition) and
``tam.model._dispatcher_gcv`` (GCV coordinate descent).
"""

import numpy as np
import torch

import tam
import tam.model._dispatcher as dispatcher
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import (
    OffsetEffect, LinearEffect, SplineEffect,
    build_phi_from_effects, build_penalty_from_effects,
)
from tam.model._dispatcher import smart_solve, smart_evaluate_rmse, smart_decompose
from tam.model._dispatcher_gcv import smart_solve_gcv


def _effects():
    return [
        OffsetEffect(lambda_p=1e-6, extrapolate="continue"),
        LinearEffect("x", scaled=1.0, lambda_p=1e-3, extrapolate="continue"),
        SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1e-3, extrapolate="continue"),
    ]


def _data(n_groups=2, n_samples=30, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(n_groups, n_samples, 1, generator=g, dtype=torch.get_default_dtype()) * 2 - 1).to(TORCH_DEVICE)
    # A smooth target so the spline model fits well.
    y = torch.sin(2.0 * x) + 0.1 * x
    return x, y


def test_smart_solve_returns_per_group_coefficients():
    effects = _effects()
    x, y = _data()
    penalty = build_penalty_from_effects(effects)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    total_k = sum(e.get_n_coeffs() for e in effects)
    coeffs = smart_solve(x, y, effects, penalty, loss, num_samples=x.shape[1])

    assert coeffs.shape == (2, total_k, 1)
    assert torch.isfinite(coeffs).all()


def test_smart_solve_fits_smooth_signal():
    """End-to-end: the solved coefficients should reconstruct the signal well."""
    effects = _effects()
    x, y = _data()
    penalty = build_penalty_from_effects(effects)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    coeffs = smart_solve(x, y, effects, penalty, loss, num_samples=x.shape[1])
    rmse = smart_evaluate_rmse(x, y, coeffs, effects)
    assert rmse.item() < 0.5  # comfortably better than predicting the mean


def test_smart_evaluate_rmse_is_nonnegative_scalar():
    effects = _effects()
    x, y = _data()
    penalty = build_penalty_from_effects(effects)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    coeffs = smart_solve(x, y, effects, penalty, loss, num_samples=x.shape[1])

    rmse = smart_evaluate_rmse(x, y, coeffs, effects)
    assert rmse.numel() == 1
    assert rmse.item() >= 0.0


def test_smart_decompose_returns_per_effect_contributions():
    effects = _effects()
    x, y = _data()
    penalty = build_penalty_from_effects(effects)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    coeffs = smart_solve(x, y, effects, penalty, loss, num_samples=x.shape[1])

    parts = smart_decompose(x, coeffs, effects)
    # offset + two effects on 'x' (collision -> prefixed) => 3 keys.
    assert len(parts) == 3
    assert "offset" in parts
    for v in parts.values():
        assert torch.isfinite(v).all()


def test_build_phi_and_penalty_dimensions_agree():
    effects = _effects()
    x, _ = _data()
    phi = build_phi_from_effects(x, effects, feature_columns=["x"])
    penalty = build_penalty_from_effects(effects)
    if penalty.is_sparse:
        penalty = penalty.to_dense()

    total_k = sum(e.get_n_coeffs() for e in effects)
    assert phi.shape[-1] == total_k
    assert penalty.shape == (total_k, total_k)


def test_smart_solve_gcv_returns_lambda_per_effect():
    effects = _effects()
    x, y = _data()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    coeffs, best_lambdas, gcv_score = smart_solve_gcv(
        x_data=x, y_data=y, effects_list=effects, loss_matrix=loss,
        alpha_p_bounds=(-6.0, 2.0), number_of_steps=3, gamma=1.4,
    )

    total_k = sum(e.get_n_coeffs() for e in effects)
    assert coeffs.shape == (2, total_k, 1)
    assert len(best_lambdas) == len(effects)
    assert np.all(np.asarray(best_lambdas) > 0)
    assert np.isfinite(gcv_score)


def test_smart_solve_gcv_with_explicit_alpha_list():
    effects = _effects()
    x, y = _data(seed=1)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    coeffs, best_lambdas, _ = smart_solve_gcv(
        x_data=x, y_data=y, effects_list=effects, loss_matrix=loss,
        alpha_p_bounds=(-6.0, 2.0), alpha_p_list=[-4.0, -2.0, 0.0],
    )
    assert len(best_lambdas) == len(effects)
    assert torch.isfinite(coeffs).all()


# --------------------------------------------------------------------------- #
# Matrix-free Conjugate Gradient routing (forced via the memory router)
# --------------------------------------------------------------------------- #

def test_cg_solver_matches_direct_solver(monkeypatch):
    """Forcing the CG route must reproduce the direct solver's coefficients."""
    x, y = _data()
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    effects_direct = _effects()
    penalty_direct = build_penalty_from_effects(effects_direct)
    coeffs_direct = smart_solve(x, y, effects_direct, penalty_direct, loss, num_samples=x.shape[1])

    # Force the dispatcher down the matrix-free CG branch.
    monkeypatch.setattr(dispatcher, "can_fit_dense_matrix", lambda *a, **k: False)
    effects_cg = _effects()
    penalty_cg = build_penalty_from_effects(effects_cg)
    coeffs_cg = smart_solve(x, y, effects_cg, penalty_cg, loss, num_samples=x.shape[1])

    assert coeffs_cg.shape == coeffs_direct.shape
    # CG converges to the same solution within its residual tolerance.
    assert torch.allclose(coeffs_cg, coeffs_direct, atol=5e-3, rtol=1e-2)


def test_cg_solver_produces_good_fit(monkeypatch):
    x, y = _data(seed=2)
    loss = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())

    monkeypatch.setattr(dispatcher, "can_fit_dense_matrix", lambda *a, **k: False)
    effects = _effects()
    penalty = build_penalty_from_effects(effects)
    coeffs = smart_solve(x, y, effects, penalty, loss, num_samples=x.shape[1])
    rmse = smart_evaluate_rmse(x, y, coeffs, effects)
    assert rmse.item() < 0.5
