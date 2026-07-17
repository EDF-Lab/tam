# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._linear``, OffsetEffect and LinearEffect."""

import torch

import tam
from tam.model.spectrum import OffsetEffect, LinearEffect


def test_offset_get_n_coeffs():
    assert OffsetEffect(lambda_p=1.0, extrapolate="continue").get_n_coeffs() == 1


def test_offset_feature_map_is_ones(normalized):
    effect = OffsetEffect(lambda_p=1e-3, extrapolate="continue")
    x = normalized(4, 12, 2)  # offset consumes the full feature tensor
    phi = effect.build_feature_map(x)
    assert phi.shape == (4, 12, 1)
    assert torch.allclose(phi, torch.ones_like(phi))


def test_offset_penalty_shape(penalty_shape):
    effect = OffsetEffect(lambda_p=2.0, extrapolate="continue")
    assert penalty_shape(effect) == (1, 1)
    assert effect.build_penalty_matrix().item() == 2.0


def test_linear_get_n_coeffs():
    assert LinearEffect("x", scaled=1.0, lambda_p=1.0, extrapolate="continue").get_n_coeffs() == 1


def test_linear_feature_map_applies_scaling(normalized):
    effect = LinearEffect("x", scaled=3.0, lambda_p=1.0, extrapolate="continue")
    x = normalized(4, 12)
    phi = effect.build_feature_map(x)
    assert phi.shape == (4, 12, 1)
    assert torch.allclose(phi.squeeze(-1), x * 3.0)


def test_linear_penalty_shape(penalty_shape):
    effect = LinearEffect("x", scaled=1.0, lambda_p=0.5, extrapolate="continue")
    assert penalty_shape(effect) == (1, 1)
    assert effect.build_penalty_matrix().item() == 0.5
