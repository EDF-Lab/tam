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


# ------------------------------------------------------------------ reproducibility: NeuralTAM draws from its own generator
def _fit_neural(data, seed=None, shuffle_split=True):
    kwargs = {} if seed is None else {"seed": seed}
    model = NeuralTAM(
        formula="load ~ l(temperature) + n(temperature, n_neurons=4)",
        group_col="smart_meter_id", date_col="timestamp",
        epochs=4, patience=4, backfit_cycles=1, val_split=0.3, shuffle_split=shuffle_split, **kwargs,
    )
    return model.fit(data)


def _forecast(model, data):
    return model.predict(data)["Estimatedload"].to_numpy()


@pytest.mark.parametrize("shuffle_split", [True, False])
def test_two_fits_agree_whatever_the_global_generator_did_in_between(dummy_panel_data, shuffle_split):
    import numpy as np
    first = _forecast(_fit_neural(dummy_panel_data, shuffle_split=shuffle_split), dummy_panel_data)
    torch.rand(1000)                                   # other code draws from the global generator
    torch.manual_seed(123)
    second = _forecast(_fit_neural(dummy_panel_data, shuffle_split=shuffle_split), dummy_panel_data)
    np.testing.assert_array_equal(first, second)


def test_a_fit_neither_reads_nor_advances_the_global_generator(dummy_panel_data):
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    _fit_neural(dummy_panel_data)
    assert torch.equal(torch.get_rng_state(), before)


def test_the_seed_changes_the_networks(dummy_panel_data):
    import numpy as np
    same_a = _forecast(_fit_neural(dummy_panel_data, seed=1), dummy_panel_data)
    same_b = _forecast(_fit_neural(dummy_panel_data, seed=1), dummy_panel_data)
    other = _forecast(_fit_neural(dummy_panel_data, seed=2), dummy_panel_data)
    np.testing.assert_array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)


def test_the_default_seed_is_42():
    assert NeuralTAM(formula="load ~ l(temperature)").seed == 42
