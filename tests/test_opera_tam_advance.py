import pytest
import numpy as np
import pandas as pd
import tam as ta

@pytest.fixture
def opera_experts_data():
    """Generates synthetic expert predictions."""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=40),
        'group': ['Group_A'] * 20 + ['Group_B'] * 20,
        'actual_target': np.random.randn(40),
        'expert_model_1': np.random.randn(40) + 0.5, # Slightly biased
        'expert_model_2': np.random.randn(40) - 0.5  # Slightly biased
    })

def test_opera_ewa_algorithm(opera_experts_data):
    """Tests the Exponentially Weighted Average (EWA) branch of the TorchScript."""
    opera = ta.OperaTAM(
        target_col='actual_target', 
        expert_cols=['expert_model_1', 'expert_model_2'], 
        algorithm='EWA', 
        eta=0.1, 
        group_col='group', 
        date_col='date'
    )
    res = opera.predict_online(opera_experts_data)
    
    assert 'prediction_opera' in res.columns
    # Ensure convex combinations hold for EWA
    weights_sum = res[['weight_expert_model_1', 'weight_expert_model_2']].sum(axis=1)
    np.testing.assert_allclose(weights_sum, 1.0, rtol=1e-5)

def test_opera_horizon_shifting(opera_experts_data):
    """Ensures horizon logic prevents target leakage by delaying weight updates."""
    # Set horizon to 3 steps forward
    opera = ta.OperaTAM(
        target_col='actual_target', 
        expert_cols=['expert_model_1', 'expert_model_2'], 
        horizon_steps=3, 
        group_col='group', 
        date_col='date'
    )
    res = opera.predict_online(opera_experts_data)
    
    # For a horizon of H, the first (H-1) steps should rely on uniform priors (1/K)
    # because the algorithm is forced to wait for the ground truth to "arrive"
    for g in ['Group_A', 'Group_B']:
        group_res = res[res['group'] == g].sort_values('date')
        
        # Check the first two steps (index 0 and 1)
        np.testing.assert_allclose(group_res.iloc[0]['weight_expert_model_1'], 0.5)
        np.testing.assert_allclose(group_res.iloc[1]['weight_expert_model_1'], 0.5)

def test_opera_absolute_loss(opera_experts_data):
    """Tests the L1 absolute loss formulation."""
    opera = ta.OperaTAM(
        target_col='actual_target', 
        expert_cols=['expert_model_1', 'expert_model_2'], 
        loss_type='absolute', 
        group_col='group', 
        date_col='date'
    )
    res = opera.predict_online(opera_experts_data)
    assert not res['prediction_opera'].isna().any(), "Absolute loss formulation returned NaNs."

def test_opera_alternate_initialization():
    """Ensures providing target/expert columns successfully generates the formula."""
    opera = ta.OperaTAM(target_col='y', expert_cols=['e1', 'e2'])
    
    assert opera.formula == "y ~ l(e1) + l(e2)", "Formula was not constructed correctly."
    assert 'e1' in opera.expert_cols, "Expert columns not extracted properly."