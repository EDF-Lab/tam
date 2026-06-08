import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def dummy_panel_data():
    """Generates a small, deterministic dummy dataset with two groups."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2022-01-01", periods=100)
    
    # Group 1
    df1 = pd.DataFrame({
        'timestamp': dates,
        'smart_meter_id': 'Meter_A',
        'temperature': rng.uniform(-5, 30, 100),
        'load': rng.normal(50, 5, 100)
    })
    
    # Group 2
    df2 = pd.DataFrame({
        'timestamp': dates,
        'smart_meter_id': 'Meter_B',
        'temperature': rng.uniform(-5, 30, 100),
        'load': rng.normal(80, 10, 100)
    })
    
    return pd.concat([df1, df2], ignore_index=True)