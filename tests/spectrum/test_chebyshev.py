# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._chebyshev.ChebyshevEffect``."""

import torch

import tam
from tam.model.spectrum import ChebyshevEffect


def test_chebyshev_contract(normalized, penalty_shape):
    effect = ChebyshevEffect("x", degree=6, s=1, lambda_p=1.0, extrapolate="continue")
    assert effect.get_n_coeffs() == 6

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 6)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (6, 6)


def test_chebyshev_first_basis_is_identity(normalized):
    """The first Chebyshev polynomial T_1(x) equals x."""
    effect = ChebyshevEffect("x", degree=3, s=0, lambda_p=1.0, extrapolate="continue")
    x = normalized(2, 8)
    phi = effect.build_feature_map(x)
    assert torch.allclose(phi[..., 0], x)
