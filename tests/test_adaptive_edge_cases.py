import pytest
import tam as ta

def test_unfitted_base_model_raises_error():
    """Ensures AdaptiveTAM protects against an unfitted base model."""
    base = ta.StaticTAM(formula="load ~ l(temperature)")
    
    with pytest.raises(ValueError, match="must be fitted before initializing AdaptiveTAM"):
        ta.AdaptiveTAM(
            base_model=base, 
            adaptive_formula="load ~ l(effect_temperature)", 
            update_interval_periods=1, 
            training_window_periods=5, 
            steps_per_period=1
        )

def test_target_mismatch_warning(dummy_panel_data):
    """Checks if the cross-target correction warning is raised."""
    base = ta.StaticTAM(formula="load ~ l(temperature)", group_col="smart_meter_id", date_col="timestamp")
    base.fit(dummy_panel_data) 
    
    with pytest.warns(UserWarning, match="does not match the base target"):
        ta.AdaptiveTAM(
            base_model=base, 
            adaptive_formula="wrong_target ~ l(effect_temperature)", 
            update_interval_periods=1, 
            training_window_periods=5, 
            steps_per_period=1
        )

def test_prepare_simulation_tensor_shapes(dummy_panel_data):
    """Verifies that the sliding window chunks data into the correct 4D/3D tensor shapes."""
    model = ta.AdaptiveTAM(
        adaptive_formula="load ~ l(temperature)", 
        update_interval_periods=1, 
        training_window_periods=5, 
        steps_per_period=1, 
        group_col="smart_meter_id", 
        date_col="timestamp"
    )
    
    model.prepare_simulation(dummy_panel_data)
    assert model.simulation_data_ is not None, "Simulation data was not generated."
    
    x_stacked, y_stacked, x_to_predict, _, _, _, _ = model.simulation_data_
    
    assert x_stacked.ndim == 4, "X training tensor must have 4 dimensions."
    assert y_stacked.ndim in [3, 4], "Y training tensor must be 3D or 4D."
    assert x_to_predict.ndim == 4, "X prediction tensor must have 4 dimensions."