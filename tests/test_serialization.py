# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

import pickle
import numpy as np
import tam as ta

def test_model_save_and_load(dummy_panel_data, tmp_path):
    """Ensures a trained model can be serialized and loaded with identical behavior."""
    model = ta.StaticTAM(formula="load ~ l(temperature)", group_col="smart_meter_id", date_col="timestamp")
    model.fit(dummy_panel_data)
    original_preds = model.predict(dummy_panel_data)
    
    # Save the model to a temporary file
    file_path = tmp_path / "tam_model.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
        
    # Load the model
    with open(file_path, "rb") as f:
        loaded_model = pickle.load(f)
        
    # Predict with loaded model
    loaded_preds = loaded_model.predict(dummy_panel_data)
    
    np.testing.assert_array_equal(original_preds, loaded_preds, "Loaded model predictions drifted.")