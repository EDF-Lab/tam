import pytest
import numpy as np
import pandas as pd
import tam as ta

def test_adaptive_tam_composition_initialization():
    """
    Ensures AdaptiveTAM strictly follows the Composition design pattern
    (owning a StaticTAM instance) rather than inheritance.
    """
    model = ta.AdaptiveTAM(
        adaptive_formula="load ~ s(temperature) + l(temperature)",
        group_col="smart_meter_id",
        date_col="timestamp",
        update_interval_periods=1,
        training_window_periods=12,
        steps_per_period=1
    )

    assert hasattr(model, 'base_model_'), "AdaptiveTAM must compose a base StaticTAM instance."
    assert model.target_col_ == "load"
    assert len(model.adaptive_model_.effects_list_) > 0

def test_adaptive_tam_online_tracking(dummy_panel_data):
    """
    Ensures the hardware-accelerated sliding window correction runs end-to-end
    without triggering OOM errors or shape mismatches.
    """
    model = ta.AdaptiveTAM(
        adaptive_formula="load ~ s(temperature, k=5) + l(temperature)",
        group_col="smart_meter_id",
        date_col="timestamp",
        update_interval_periods=1,
        training_window_periods=20,
        steps_per_period=1
    )

    predictions = model.predict_online(dummy_panel_data)

    assert predictions is not None, "predict_online returned None."
    assert len(predictions) == len(dummy_panel_data), "Prediction length mismatch."
    adapted_col = f"AdaptedEstimated{model.target_col_}"
    assert adapted_col in predictions.columns, f"Missing column '{adapted_col}' in output."

def test_adaptive_tam_coordinate_descent(dummy_panel_data):
    """
    Tests AdaptiveTAM grid search with a token formula to ensure coordinate
    descent selects a valid configuration.
    """
    model = ta.AdaptiveTAM(
        adaptive_formula="load ~ s(temperature, k='grid_k')",
        group_col="smart_meter_id",
        date_col="timestamp",
        update_interval_periods=1,
        training_window_periods=10,
        steps_per_period=1
    )

    best_model = model.grid_search_fit(
        data_val=dummy_panel_data,
        grid_search_config={'grid_k': [5, 10]}
    )

    assert best_model is not None, "Grid search returned None."
    assert isinstance(best_model, ta.AdaptiveTAM), "Grid search must return an AdaptiveTAM instance."
    assert best_model.predictions_ is not None, "Grid search model did not run simulation."
