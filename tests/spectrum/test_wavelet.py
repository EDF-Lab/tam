# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._wavelet.WaveletEffect`` (Ricker wavelets)."""

import torch

import tam
from tam.model.spectrum import WaveletEffect


def test_wavelet_contract(normalized, penalty_shape):
    effect = WaveletEffect("x", n_scales=3, n_locations=4, lambda_p=1.0, extrapolate="continue")
    assert effect.get_n_coeffs() == 12  # n_scales * n_locations

    phi = effect.build_feature_map(normalized(4, 12))
    assert phi.shape == (4, 12, 12)
    assert torch.isfinite(phi).all()

    assert penalty_shape(effect) == (12, 12)


def test_wavelet_grid_is_cached_after_first_call(normalized):
    effect = WaveletEffect("x", n_scales=2, n_locations=3, lambda_p=1.0, extrapolate="continue")
    effect.build_feature_map(normalized(4, 12))
    locations = effect.locations
    assert locations is not None
    # A second call reuses the fitted time-scale grid rather than re-deriving it.
    effect.build_feature_map(normalized(2, 6, seed=55))
    assert effect.locations is locations
