# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

import numpy as np
import pandas as pd
import tam as ta

def test_spline_ood_extrapolation(dummy_panel_data):
    """Ensures the model doesn't crash when predicting wildly outside training bounds."""
    model = ta.StaticTAM(
        formula="load ~ s(temperature, extrapolate='saturation')",
        group_col="smart_meter_id",
        date_col="timestamp"
    )
    model.fit(dummy_panel_data)
    
    # Create extreme OOD data (e.g., temperature = 5000 degrees)
    ood_data = pd.DataFrame({
        'timestamp': pd.date_range(start="2023-01-01", periods=10),
        'smart_meter_id': 'Meter_A',
        'temperature': np.array([5000.0] * 10) 
    })
    
    # It should predict safely without throwing NaN or crashing PyTorch
    preds = model.predict(ood_data)
    est_col = f"Estimated{model.target_col_}"
    assert not preds[est_col].isna().any(), "Extrapolation resulted in NaNs."
    # Since it's 'saturation', variance of predictions should be 0 (it flatlined)
    assert np.var(preds[est_col].values) < 1e-5, "Saturation extrapolation failed to flatline."