# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.neural``, the Deep-GAM hybrid (NeuralTAM) and its
``DeepNeuralComponent`` building block.

Networks are kept tiny (few neurons, 2 epochs) so the backfitting loop runs fast.
"""

import numpy as np
import pandas as pd
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
    same_a = _forecast(_fit_neural(dummy_panel_data, seed=1), dummy_panel_data)
    same_b = _forecast(_fit_neural(dummy_panel_data, seed=1), dummy_panel_data)
    other = _forecast(_fit_neural(dummy_panel_data, seed=2), dummy_panel_data)
    np.testing.assert_array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)


def test_the_default_seed_is_42():
    assert NeuralTAM(formula="load ~ l(temperature)").seed == 42


# ------------------------------------------------------------------ a NeuralTAM decomposes and serves as a base model like a StaticTAM
NEURAL_FORMULA = "load ~ l(temperature) + s(temperature, k=5) + n(temperature, n_neurons=4)"


def _pair(data):
    static = ta.StaticTAM(formula=NEURAL_FORMULA, group_col="smart_meter_id", date_col="timestamp").fit(data)
    neural = NeuralTAM(formula=NEURAL_FORMULA, group_col="smart_meter_id", date_col="timestamp",
                       epochs=4, patience=4, backfit_cycles=1).fit(data)
    return static, neural


def test_neural_decomposition_has_the_columns_of_the_static_one_and_sums_to_the_forecast(dummy_panel_data):
    static, neural = _pair(dummy_panel_data)
    d_static, d_neural = static.decompose_prediction(dummy_panel_data), neural.decompose_prediction(dummy_panel_data)
    effects = lambda frame: [c for c in frame.columns if c.startswith("effect_")]
    assert effects(d_neural) == effects(d_static)
    np.testing.assert_allclose(d_neural[effects(d_neural)].sum(axis=1), neural.predict(dummy_panel_data)["Estimatedload"], rtol=1e-12)


def test_neural_decomposition_places_effects_by_row_whatever_the_row_order_and_labels(dummy_panel_data):
    _, neural = _pair(dummy_panel_data)
    reference = neural.decompose_prediction(dummy_panel_data)
    shuffled = dummy_panel_data.sample(frac=1.0, random_state=3)
    duplicated = pd.concat([dummy_panel_data.iloc[:50], dummy_panel_data.iloc[50:]])        # same labels, concatenated
    effects = [c for c in reference.columns if c.startswith("effect_")]
    got = neural.decompose_prediction(shuffled)
    np.testing.assert_allclose(got[effects].to_numpy(), reference.loc[shuffled.index, effects].to_numpy(), rtol=1e-12)
    got = neural.decompose_prediction(duplicated.reset_index(drop=True).set_axis(np.zeros(len(duplicated), dtype=int)))
    np.testing.assert_allclose(got[effects].to_numpy(), reference[effects].to_numpy(), rtol=1e-12)


def test_an_adaptive_model_runs_behind_a_neural_tam_like_behind_a_static_tam(dummy_panel_data):
    static, neural = _pair(dummy_panel_data)
    outputs = {}
    for name, base in (("static", static), ("neural", neural)):
        sim = dummy_panel_data.copy()
        sim["L_Res"] = (sim["load"] - base.predict(sim)["Estimatedload"]).groupby(sim["smart_meter_id"]).shift(1)
        sim = sim.dropna(subset=["L_Res"]).reset_index(drop=True)
        adaptive = ta.AdaptiveTAM(base_model=base, adaptive_formula="Residualload ~ l(L_Res)", update_interval_periods=5,
                                  training_window_periods=30, steps_per_period=1, horizon_steps=1)
        outputs[name] = adaptive.predict_online(sim)
        assert adaptive.predict(sim)["AdaptedEstimatedload"].notna().all()
    effects = lambda frame: [c for c in frame.columns if c.startswith("effect_") or c.startswith("Adapted")]
    assert effects(outputs["neural"]) == effects(outputs["static"])
    assert len(outputs["neural"]) == len(outputs["static"])
