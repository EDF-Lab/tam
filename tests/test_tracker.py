import numpy as np
import pandas as pd
from tam.evaluation.tracker import BenchmarkTracker

def test_tracker_nan_safety():
    """Ensures evaluation metrics do not crash when given missing target data."""
    tracker = BenchmarkTracker(model_name="TestTAM")

    y_true = np.array([100.0, 110.0, np.nan, 105.0, 120.0])
    y_pred = np.array([102.0, 108.0, 115.0, 106.0, 119.0])

    # Assign predictions and trigger metric computation via slice_and_evaluate
    tracker.y_pred_full = y_pred
    df_test = pd.DataFrame({'value': y_true})
    tracker.slice_and_evaluate({'test': df_test}, target_col='value')

    rmse = tracker.get_metric('test', 'RMSE')
    mape = tracker.get_metric('test', 'MAPE')

    assert not np.isnan(rmse), "RMSE calculation failed on NaN-containing targets."
    assert not np.isnan(mape), "MAPE calculation failed on NaN-containing targets."
