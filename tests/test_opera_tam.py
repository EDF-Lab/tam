import pytest
import numpy as np
import pandas as pd
import tam as ta

def test_opera_aggregation():
    """Tests the MLpol routing logic of OperaTAM on dummy experts."""
    # Create a dummy dataframe with 2 "expert" models and a target
    df = pd.DataFrame({
        'timestamp': pd.date_range(start="2022-01-01", periods=50),
        'target': np.random.randn(50),
        'expert_1': np.random.randn(50),
        'expert_2': np.random.randn(50)
    })
    
    # Initialize OperaTAM
    opera = ta.OperaTAM(
        target_col='target',
        expert_cols=['expert_1', 'expert_2'],
        algorithm='MLpol'
    )
    
    # Test online tracking
    aggregated_preds = opera.predict_online(df)
    
    # Ensure weights were calculated and predictions match the length
    assert len(aggregated_preds) == len(df)
    assert hasattr(opera, 'weights_history_'), "OperaTAM did not save the historical weights."
    assert len(opera.weights_history_) > 0, "weights_history_ is empty after predict_online."

    # Ensure weights sum to 1 at each time step (convex combination) for each group
    for group_weights in opera.weights_history_.values():
        weights_sum = np.sum(group_weights, axis=1)
        np.testing.assert_allclose(weights_sum, 1.0, rtol=1e-5)