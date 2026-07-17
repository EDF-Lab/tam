# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.neural``, the Deep-GAM hybrid (NeuralTAM) and its
``DeepNeuralComponent`` building block.

Networks are kept tiny (few neurons, 2 epochs) so the backfitting loop runs fast.
"""

import torch
import pytest

import tam as ta
from tam.model.neural import NeuralTAM, DeepNeuralComponent


@pytest.mark.parametrize("activation", ["relu", "tanh", "cos", "unknown_defaults_to_relu"])
def test_deep_neural_component_forward(activation):
    net = DeepNeuralComponent(input_dim=3, n_neurons=4, n_hidden_layers=2, activation_name=activation)
    x = torch.randn(5, 3, dtype=torch.get_default_dtype())
    out = net(x)
    assert out.shape == (5, 1)
    # The final layer is zero-initialised, so the untrained net outputs zeros.
    assert torch.allclose(out, torch.zeros_like(out))


def test_neural_tam_passthrough_properties():
    model = NeuralTAM(formula="load ~ l(temperature)")
    # The properties proxy through to the base StaticTAM, which builds its
    # effects eagerly at construction time.
    assert model.effects_list_ is model.base_additive_model.effects_list_
    assert len(model.effects_list_) > 0
    assert isinstance(model.features_config_, dict)
    assert "features" in model.features_config_


def test_neural_tam_fit_without_neural_effect_skips_backfit(dummy_panel_data):
    """A purely structured formula trains the base GAM and skips MLP backfitting."""
    model = NeuralTAM(
        formula="load ~ s(temperature, k=5) + l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
        epochs=2,
    )
    model.fit(dummy_panel_data)

    assert model.coefficients_ is not None
    assert model.target_col_ == "load"
    # No neural effects => no MLPs were trained.
    assert all(len(v) == 0 for v in model.mlps_.values()) or model.mlps_ == {}

    preds = model.predict(dummy_panel_data)
    est_col = "Estimatedload"
    assert est_col in preds.columns
    assert len(preds) == len(dummy_panel_data)


def test_neural_tam_fit_with_neural_effect_backfits(dummy_panel_data):
    """An n() term triggers the per-group MLP backfitting path."""
    model = NeuralTAM(
        formula="load ~ l(temperature) + n(temperature, n_neurons=4)",
        group_col="smart_meter_id", date_col="timestamp",
        epochs=2, patience=2, backfit_cycles=1,
    )
    model.fit(dummy_panel_data)

    assert model.coefficients_ is not None
    # At least one group trained an MLP for the neural feature.
    assert any(len(group_mlps) > 0 for group_mlps in model.mlps_.values())

    preds = model.predict(dummy_panel_data)
    assert "Estimatedload" in preds.columns
    assert len(preds) == len(dummy_panel_data)
