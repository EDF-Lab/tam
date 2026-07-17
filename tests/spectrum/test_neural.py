# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""Tests for ``tam.model.spectrum._neural.NeuralEffect`` (NEPT projection)."""

import pytest
import torch

import tam
from tam.common.utils import TORCH_DEVICE
from tam.model.spectrum import NeuralEffect


def _effect(activation="relu", n_neurons=8, additional=None, layers=1, seed=42):
    return NeuralEffect(
        "x", n_neurons=n_neurons, activation=activation, lambda_p=1.0,
        additional_features=additional, seed=seed, n_hidden_layers=layers,
        extrapolate="continue",
    )


def test_neural_contract(normalized, penalty_shape):
    effect = _effect(n_neurons=8, layers=2)
    assert effect.get_n_coeffs() == 8

    phi = effect.build_feature_map(normalized(4, 12, 1))
    assert phi.shape == (4, 12, 8)
    assert penalty_shape(effect) == (8, 8)


@pytest.mark.parametrize("activation", ["relu", "cos", "tanh"])
def test_neural_activations_are_finite(activation, normalized):
    phi = _effect(activation=activation, n_neurons=6).build_feature_map(normalized(2, 5, 1))
    assert phi.shape == (2, 5, 6)
    assert torch.isfinite(phi).all()


def test_neural_invalid_activation_raises(normalized):
    with pytest.raises(ValueError, match="Unknown activation"):
        _effect(activation="sigmoid", n_neurons=4).build_feature_map(normalized(2, 5, 1))


def test_neural_weights_are_seed_reproducible(normalized):
    x = normalized(3, 6, 1)
    a = _effect(activation="tanh", seed=99).build_feature_map(x)
    b = _effect(activation="tanh", seed=99).build_feature_map(x)
    assert torch.allclose(a, b)


def test_neural_additional_features_recorded():
    effect = _effect(additional=["humidity", "wind"])
    assert effect.input_features == ["x", "humidity", "wind"]


def test_neural_two_dimensional_input_routing():
    """A 2D input whose last axis matches the feature count is consumed as-is."""
    effect = _effect(additional=["lon"], n_neurons=5)  # input_features = ['x', 'lon']
    x = torch.rand(7, 2, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) * 2 - 1
    phi = effect.build_feature_map(x)
    assert phi.shape == (7, 5)


def test_neural_two_dimensional_tensor_product_routing(normalized):
    """A 2D input whose last axis is the time dimension (te path) is reshaped."""
    effect = _effect(n_neurons=5)  # univariate: input_features = ['x']
    x = normalized(4, 12)  # last axis (12) != feature count (1)
    phi = effect.build_feature_map(x)
    assert phi.shape == (4, 12, 5)


def test_neural_memory_probe_shape():
    """The framework's (1, 1) VRAM probe must return without crashing."""
    effect = _effect(n_neurons=4)
    probe = torch.zeros(1, 1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    phi = effect.build_feature_map(probe)
    assert phi.shape[-1] == 4
