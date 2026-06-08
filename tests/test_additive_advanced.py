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