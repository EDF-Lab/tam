# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._tensor.TensorProductEffect`` (interactions)."""

import torch

import tam
from tam.model.spectrum import TensorProductEffect, SplineEffect


def _two_splines():
    a = SplineEffect("a", n_knots=6, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    b = SplineEffect("b", n_knots=5, spline_degree=3, penalty_order=2, lambda_p=1.0, extrapolate="continue")
    return a, b


def test_tensor_product_contract(normalized, penalty_shape):
    a, b = _two_splines()
    effect = TensorProductEffect([a, b], lambda_p=1.0, extrapolate="continue")
    expected = a.get_n_coeffs() * b.get_n_coeffs()  # 9 * 8
    assert effect.get_n_coeffs() == expected

    phi = effect.build_feature_map(normalized(4, 12, 2))
    assert phi.shape == (4, 12, expected)
    assert penalty_shape(effect) == (expected, expected)


def test_tensor_product_name_composition():
    a, b = _two_splines()
    effect = TensorProductEffect([a, b], lambda_p=1.0, extrapolate="continue")
    assert effect.feature_name == "te_a_x_b"


def test_kronecker_product_helper():
    t1 = torch.tensor([[1.0, 2.0]])
    t2 = torch.tensor([[3.0, 4.0, 5.0]])
    out = TensorProductEffect.kronecker_product_einsum(t1, t2)
    assert out.shape == (1, 6)
    assert torch.allclose(out, torch.tensor([[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]]))


def test_tensor_product_penalty_is_anisotropic_sum(penalty_shape):
    """The penalty sums each direction expanded by identities on the others."""
    a, b = _two_splines()
    effect = TensorProductEffect([a, b], lambda_p=1.0, extrapolate="continue")
    P = effect.build_penalty_matrix()
    if P.is_sparse:
        P = P.to_dense()
    assert P.shape == (a.get_n_coeffs() * b.get_n_coeffs(),) * 2
    # The anisotropic penalty is symmetric.
    assert torch.allclose(P, P.T)
