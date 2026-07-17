# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

import numpy as np
import pytest
import tam as ta

def test_static_tam_initialization():
    """Ensures the model initializes and parses the formula correctly."""
    model = ta.StaticTAM(formula="load ~ s(temperature) + l(temperature)")

    # Check that the formula parser extracted the target
    assert model.target_col_ == "load"
    assert len(model.effects_list_) > 0

def test_static_tam_fit_predict(dummy_panel_data):
    """Ensures the Primal exact solver runs end-to-end without OOM or shape errors."""
    model = ta.StaticTAM(
        formula="load ~ s(temperature, k=5) + l(temperature)",
        group_col="smart_meter_id",
        date_col="timestamp"
    )
    
    # 1. Test fitting (Should use exact GCV or direct solver)
    model.fit(dummy_panel_data)
    
    # 2. Test prediction
    predictions = model.predict(dummy_panel_data)
    
    # 3. Assertions
    assert predictions is not None, "Predict returned None."
    assert len(predictions) == len(dummy_panel_data), "Prediction length mismatch."
    # Ensure it didn't just return NaNs
    assert not predictions.isna().all().any(), "Predictions contain all NaNs."

def test_static_tam_fit_is_deterministic(dummy_panel_data):
    """Fitting the same model twice must produce identical predictions."""
    formula = "load ~ s(temperature, k=5) + l(temperature)"

    model_a = ta.StaticTAM(formula=formula, group_col="smart_meter_id", date_col="timestamp")
    model_a.fit(dummy_panel_data)
    preds_a = model_a.predict(dummy_panel_data)

    model_b = ta.StaticTAM(formula=formula, group_col="smart_meter_id", date_col="timestamp")
    model_b.fit(dummy_panel_data)
    preds_b = model_b.predict(dummy_panel_data)

    est_col = f"Estimated{model_a.target_col_}"
    np.testing.assert_allclose(
        preds_a[est_col].values, preds_b[est_col].values, rtol=1e-5, atol=1e-5
    )