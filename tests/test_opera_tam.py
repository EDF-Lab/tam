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

def test_opera_fit_predict_operational():
    """
    Tests the production inference API for OperaTAM.
    Ensures weights are frozen and applied successfully to target-less future data.
    """
    # Create historical data
    train_df = pd.DataFrame({
        'timestamp': pd.date_range(start="2022-01-01", periods=50),
        'target': np.random.randn(50),
        'expert_1': np.random.randn(50),
        'expert_2': np.random.randn(50)
    })
    
    opera = ta.OperaTAM(
        target_col='target',
        expert_cols=['expert_1', 'expert_2'],
        algorithm='MLpol'
    )
    
    # 1. Historical Simulation
    opera.fit(train_df)
    
    assert hasattr(opera, 'weights_history_')
    assert len(opera.weights_history_) > 0
    
    # 2. Create Future Data (No target column)
    future_df = pd.DataFrame({
        'timestamp': pd.date_range(start="2022-02-20", periods=10),
        'expert_1': np.random.randn(10),
        'expert_2': np.random.randn(10)
    })
    
    preds = opera.predict(future_df)
    
    assert "prediction_opera" in preds.columns
    assert "weight_expert_1" in preds.columns
    assert len(preds) == len(future_df)
    assert not preds["prediction_opera"].isna().any()

def test_opera_coherence():
    """
    Parity Test: Ensures the final step of the dynamic expert aggregation 
    matches exactly with the frozen weights applied via predict().
    """
    df = pd.DataFrame({
        'timestamp': pd.date_range(start="2022-01-01", periods=50),
        'target': np.random.randn(50),
        'expert_1': np.random.randn(50),
        'expert_2': np.random.randn(50)
    })
    
    # 1. Dynamic continuous aggregation
    model_dyn = ta.OperaTAM(
        target_col='target', expert_cols=['expert_1', 'expert_2'], 
        algorithm='MLpol', horizon_steps=1
    )
    res_dyn = model_dyn.predict_online(df)
    
    # 2. Frozen state inference
    model_stat = ta.OperaTAM(
        target_col='target', expert_cols=['expert_1', 'expert_2'], 
        algorithm='MLpol', horizon_steps=1
    )
    model_stat.fit(df)
    res_stat = model_stat.predict(df)
    
    # 3. Mathematical Parity Check on the last timestamp
    last_dyn = res_dyn.iloc[-1]['prediction_opera']
    last_stat = res_stat.iloc[-1]['prediction_opera']
    
    np.testing.assert_allclose(
        last_dyn, last_stat, rtol=1e-4, 
        err_msg="OperaTAM frozen predict() diverged from the final state of predict_online()."
    )