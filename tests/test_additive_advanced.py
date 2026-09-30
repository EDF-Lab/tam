# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

import pytest
import numpy as np
import pandas as pd
import tam as ta

def test_decompose_prediction_sum(dummy_panel_data):
    """Ensures the sum of decomposed effects equals the final prediction."""
    model = ta.StaticTAM(
        formula="load ~ s(temperature) + l(temperature)", 
        group_col="smart_meter_id", 
        date_col="timestamp"
    )
    model.fit(dummy_panel_data)
    
    preds = model.predict(dummy_panel_data)
    decomp = model.decompose_prediction(dummy_panel_data)

    effect_cols = [col for col in decomp.columns if col.startswith('effect_')]

    assert len(effect_cols) > 0, "Decomposition failed to generate effect columns."

    est_col = f"Estimated{model.target_col_}"
    assert est_col in preds.columns, f"Prediction dataframe is missing the '{est_col}' column."

    summed_effects = decomp[effect_cols].sum(axis=1)
    estimated_values = preds[est_col]

    correlation = np.corrcoef(summed_effects, estimated_values)[0, 1]
    assert correlation > 0.99, "Decomposed effects do not sum up to the total prediction."

def test_auto_fit_updates_lambda(dummy_panel_data):
    """Verifies that GCV actually modifies the lambda_p attributes of the effects."""
    model = ta.StaticTAM(
        formula="load ~ s(temperature) + l(temperature)", 
        group_col="smart_meter_id", 
        date_col="timestamp"
    )
    
    model.auto_fit(dummy_panel_data, alpha_p_bounds=(-5.0, 2.0), number_of_steps=3)
    
    for effect in model.effects_list_:
        assert hasattr(effect, "lambda_p"), f"Effect {effect.name} is missing lambda_p."
        assert effect.lambda_p > 0, f"Penalty for {effect.name} must be > 0, got {effect.lambda_p}"

@pytest.mark.parametrize("formula", [
    "load ~ s(temperature, ap=-6.0) + l(temperature, ap=-6.0)",
    # A sparsity-adaptive tree builds its penalty from the leaf counts the transform
    # leaves behind, so GCV is the first caller to assemble it after a transform.
    "load ~ t(temperature, n_trees=2, max_depth=2, sp_alpha=1.0)",
])
def test_auto_fit_lambdas_reproduce_the_auto_fit(formula, dummy_panel_data):
    """A model refitted with the lambdas GCV stored must return the GCV model.

    ``auto_fit()`` then ``fit()`` is the ordinary sklearn contract, and the path
    AutoTAM takes when it refits a champion on the full training window. The
    spline formula pins a small ap so the check fails if the search penalty and the
    stored penalty ever drift apart.
    """
    model = ta.StaticTAM(
        formula=formula,
        group_col="smart_meter_id",
        date_col="timestamp"
    )

    model.auto_fit(dummy_panel_data, alpha_p_bounds=(-5.0, 2.0), number_of_steps=3)
    est_col = f"Estimated{model.target_col_}"
    auto_pred = model.predict(dummy_panel_data)[est_col].to_numpy()
    auto_lambdas = [effect.lambda_p for effect in model.effects_list_]

    model.fit(dummy_panel_data)
    refit_pred = model.predict(dummy_panel_data)[est_col].to_numpy()

    assert [effect.lambda_p for effect in model.effects_list_] == auto_lambdas
    np.testing.assert_allclose(refit_pred, auto_pred, rtol=1e-6, atol=1e-6)

def test_static_grid_search(dummy_panel_data):
    """Tests that coordinate descent works directly on the static model."""
    model = ta.StaticTAM(
        formula="load ~ s(temperature, k='grid_k')", 
        group_col="smart_meter_id", 
        date_col="timestamp"
    )
    
    config = {'grid_k': [5, 10]}
    
    best_model = model.grid_search_fit(
        data_train=dummy_panel_data, 
        data_val=dummy_panel_data, 
        grid_search_config=config
    )
    
    assert best_model.coefficients_ is not None, "Grid search model did not fit."

def test_summary_dataframe(dummy_panel_data):
    """Ensures the architecture summary builds correctly."""
    model = ta.StaticTAM(formula="load ~ s(temperature)")
    model.fit(dummy_panel_data)
    
    summary_df = model.summary()
    assert not summary_df.empty, "Summary DataFrame is empty."
    assert "Feature" in summary_df.columns
    assert "Complexity (D)" in summary_df.columns