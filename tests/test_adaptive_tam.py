# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

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

def test_adaptive_tam_fit_predict_operational(dummy_panel_data):
    """
    Tests the new O(1) inference API for production deployment.
    Ensures fit() saves the final state and predict() applies it flawlessly,
    even when the target column is physically missing from the inference data.
    """
    model = ta.AdaptiveTAM(
        adaptive_formula="load ~ l(temperature)",
        group_col="smart_meter_id",
        date_col="timestamp",
        update_interval_periods=1,
        training_window_periods=10,
        steps_per_period=1
    )
    
    # 1. Historical Simulation
    model.fit(dummy_panel_data)
    
    assert hasattr(model, 'last_state_dict_'), "fit() did not create last_state_dict_."
    assert model.last_state_dict_ is not None, "Final state was not saved."
    assert set(model.norm_params_) == set(model.last_state_dict_), "fit() did not save the normalisation of the final window."
    
    # 2. Operational Inference (Drop the target column entirely)
    prod_data = dummy_panel_data.drop(columns=["load"])
    
    predictions = model.predict(prod_data)
    
    assert "AdaptedEstimatedload" in predictions.columns, "Missing adapted prediction column."
    assert len(predictions) == len(prod_data), "Prediction length mismatch."
    assert not predictions["AdaptedEstimatedload"].isna().any(), "Predictions contain NaNs."

def test_adaptive_tam_coherence(dummy_panel_data):
    """
    Operational parity: forecasting the next day from fit(history) gives the forecast of the online simulation for that day
    (update every period, horizon 1): the frozen state is the window that ends at the last observed row.
    """
    def make():
        return ta.AdaptiveTAM(
            adaptive_formula="load ~ l(temperature)", group_col="smart_meter_id", date_col="timestamp",
            update_interval_periods=1, training_window_periods=5, steps_per_period=1, horizon_steps=1
        )

    # 1. Dynamic continuous simulation over every day
    res_dyn = make().predict_online(dummy_panel_data)

    # 2. Frozen state: fit on the history up to the day before the last one, forecast the last day
    last_day = dummy_panel_data['timestamp'].max()
    history = dummy_panel_data[dummy_panel_data['timestamp'] < last_day]
    tomorrow = dummy_panel_data[dummy_panel_data['timestamp'] == last_day].drop(columns=["load"])
    res_stat = make().fit(history).predict(tomorrow)

    # 3. Same forecast for every group
    dyn = res_dyn[res_dyn['timestamp'] == last_day].set_index('smart_meter_id')['AdaptedEstimatedload']
    stat = res_stat.set_index('smart_meter_id')['AdaptedEstimatedload']
    np.testing.assert_allclose(
        stat.loc[dyn.index].to_numpy(), dyn.to_numpy(), rtol=1e-8,
        err_msg="AdaptiveTAM fit() then predict() diverged from the online simulation."
    )
