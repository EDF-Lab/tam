# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""Tests for ``tam.model.spectrum._spline.SplineEffect`` (P-splines)."""

import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import SplineEffect


def test_spline_contract(normalized, penalty_shape):
    effect = SplineEffect("x", n_knots=10, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    assert effect.get_n_coeffs() == 13  # n_knots + spline_degree

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 13)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (13, 13)


def test_spline_zero_penalty_order_uses_identity(penalty_shape):
    """penalty_order=0 reduces the difference operator to the identity."""
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=0, lambda_p=2.0, extrapolate="continue")
    k = effect.get_n_coeffs()
    P = effect.build_penalty_matrix()
    assert tuple(P.shape) == (k, k)
    # lambda_p * (I^T I) = lambda_p * I
    assert torch.allclose(P, 2.0 * torch.eye(k, device=TORCH_DEVICE, dtype=torch.get_default_dtype()))


def test_spline_knots_handle_degenerate_input():
    """All-non-finite input must fall back to a default [0, 1] knot span."""
    effect = SplineEffect("x", n_knots=6, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    degenerate = torch.full((20,), float("nan"), device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    knots = effect._get_knots(degenerate, is_dummy=True)
    assert torch.isfinite(knots).all()


def test_spline_knots_widen_for_constant_input():
    """Constant data (x_min == x_max) is widened to avoid a zero-width knot span."""
    effect = SplineEffect("x", n_knots=6, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    constant = torch.full((10,), 0.5, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    knots = effect._get_knots(constant, is_dummy=True)
    assert torch.isfinite(knots).all()
    assert knots.min() < knots.max()


def test_spline_knot_cache_reused(normalized):
    effect = SplineEffect("x", n_knots=8, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    effect.build_feature_map(normalized(4, 12))
    cached = effect._cached_knots
    assert cached is not None
    # A second call must reuse the cached knot vector rather than recomputing it.
    effect.build_feature_map(normalized(2, 6, seed=999))
    assert effect._cached_knots is cached


def test_spline_bsplvb_active_span_matches_dense_and_ood_scale():
    effect = SplineEffect("x", n_knots=18, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="linear")
    x = torch.linspace(-1.0, 1.0, 15000, device=TORCH_DEVICE, dtype=torch.get_default_dtype()).unsqueeze(0)
    knots = effect._get_knots(x).to(device=x.device, dtype=x.dtype)
    assert x.numel() * knots.numel() >= 262144

    via_dispatch = effect._cox_de_boor(x, knots, 3)
    x_exp = x.unsqueeze(-1)
    x_c = torch.clamp(x_exp, min=knots[3], max=knots[-4] - 1e-6)
    dense = ((x_c >= knots[:-1]) & (x_c < knots[1:])).to(x.dtype)
    for d in range(1, 4):
        dt_l = knots[d:-1] - knots[:-d - 1]
        inv_l = torch.where(dt_l > 1e-8, 1.0 / dt_l.clamp_min(1e-8), torch.zeros_like(dt_l))
        dt_r = knots[d + 1:] - knots[1:-d]
        inv_r = torch.where(dt_r > 1e-8, 1.0 / dt_r.clamp_min(1e-8), torch.zeros_like(dt_r))
        dense = ((x_c - knots[:-d - 1]) * inv_l) * dense[..., :-1] + ((knots[d + 1:] - x_c) * inv_r) * dense[..., 1:]

    assert torch.allclose(via_dispatch, dense, atol=1e-12, rtol=1e-12)
    assert effect._ood_slope_scale() == 1.0
    effect._reml_fitted = True
    assert abs(effect._ood_slope_scale() - 2.0 / 3.0) < 1e-12
    linear_ord = SplineEffect("x", n_knots=8, spline_degree=1, penalty_order=1, lambda_p=1.0, extrapolate="linear")
    linear_ord._reml_fitted = True
    assert linear_ord._ood_slope_scale() == 1.0
