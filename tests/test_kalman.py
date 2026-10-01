# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

r"""
Unit tests for ``tam.model.kalman.KalmanTAM``, the Block Kalman Filter
meta-learner that tracks drifting coefficients online.
"""

import warnings
import numpy as np
import pytest

import tam as ta
from tam.model.kalman import KalmanTAM

# These tests use small panels, an explicit block_size and the default reference period: the messages are expected.
pytestmark = [
    pytest.mark.filterwarnings("ignore:KalmanTAM:UserWarning"),
    pytest.mark.filterwarnings("ignore:The calibration period includes:UserWarning"),
    pytest.mark.filterwarnings("ignore:lookback_days is deprecated:FutureWarning"),
]


def test_kalman_standalone_predict_online(dummy_panel_data):
    """Without a base model, the filter tracks the target directly from 0."""
    model = KalmanTAM(
        kalman_formula="load ~ l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
        block_size=16,
    )
    out = model.predict_online(dummy_panel_data)

    col = "KalmanAdapted_load"
    assert col in out.columns
    assert len(out) == len(dummy_panel_data)
    assert not out[col].isna().all()
    assert model.states_history_ is not None


def test_kalman_unfitted_base_model_raises():
    base = ta.StaticTAM(formula="load ~ l(temperature)")  # never fitted
    with pytest.raises(ValueError, match="must be fitted"):
        KalmanTAM(kalman_formula="load ~ l(effect_temperature)", base_model=base)


def test_kalman_target_mismatch_warns(dummy_panel_data):
    base = ta.StaticTAM(
        formula="load ~ l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
    )
    base.fit(dummy_panel_data)
    with pytest.warns(UserWarning, match="does not match base model target"):
        KalmanTAM(
            kalman_formula="other ~ l(temperature)",
            base_model=base,
            use_decomposition=False,
        )


def test_kalman_with_fitted_base_model_residual_tracking(dummy_panel_data):
    base = ta.StaticTAM(
        formula="load ~ s(temperature, k=5)",
        group_col="smart_meter_id", date_col="timestamp",
    )
    base.fit(dummy_panel_data)

    model = KalmanTAM(
        kalman_formula="load ~ l(temperature)",
        base_model=base,
        group_col="smart_meter_id", date_col="timestamp",
        use_decomposition=False,
        block_size=16,
    )
    out = model.predict_online(dummy_panel_data)
    assert "KalmanAdapted_load" in out.columns
    assert len(out) == len(dummy_panel_data)


def test_kalman_horizon_shifting(dummy_panel_data):
    """horizon_steps > 1 forces single-step blocks and shifts the state forward."""
    model = KalmanTAM(
        kalman_formula="load ~ l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
        horizon_steps=3,
    )
    out = model.predict_online(dummy_panel_data)
    assert "KalmanAdapted_load" in out.columns
    assert len(out) == len(dummy_panel_data)


def test_kalman_tune_hyperparameters(dummy_panel_data):
    model = KalmanTAM(
        kalman_formula="load ~ l(temperature)",
        group_col="smart_meter_id", date_col="timestamp",
        block_size=16,
    )
    best_params, best_rmse = model.tune_hyperparameters(
        dummy_panel_data,
        param_grid={
            "observation_noise_var": [0.5, 1.0],
            "process_noise_var": [1e-4, 1e-3],
        },
        lookback_days=5,
    )
    assert isinstance(best_params, dict)
    assert "observation_noise_var" in best_params
    assert np.isfinite(best_rmse)

def test_kalman_fit_predict_operational(dummy_panel_data):
    """
    Tests the frozen inference API for KalmanTAM.
    Verifies that the patched feature extractor successfully bypasses target 
    extraction during out-of-sample prediction.
    """
    model = KalmanTAM(
        kalman_formula="load ~ l(temperature)",
        group_col="smart_meter_id", 
        date_col="timestamp",
        block_size=16,
    )
    
    # 1. Historical Tracking
    model.fit(dummy_panel_data)
    
    assert hasattr(model, 'last_state_dict_'), "fit() did not save Kalman states."
    assert hasattr(model, 'scale_dict_'), "fit() did not save Kalman scaling factors."
    assert len(model.last_state_dict_) > 0
    
    # 2. Operational Inference (Drop the target column)
    prod_data = dummy_panel_data.drop(columns=["load"])
    
    preds = model.predict(prod_data)
    
    assert "KalmanAdapted_load" in preds.columns
    assert len(preds) == len(prod_data)
    assert not preds["KalmanAdapted_load"].isna().any()

def test_kalman_coherence(dummy_panel_data):
    """
    Parity Test: the frozen state of ``fit()`` forecasts the next step exactly as the online simulation does
    (``block_size=1``): fit on all but the last day, predict the last day.
    """
    dates = np.sort(dummy_panel_data["timestamp"].unique())
    history = dummy_panel_data[dummy_panel_data["timestamp"] < dates[-1]]
    last_day = dummy_panel_data[dummy_panel_data["timestamp"] == dates[-1]]

    kwargs = dict(kalman_formula="load ~ l(temperature)", group_col="smart_meter_id", date_col="timestamp",
                  block_size=1, horizon_steps=1, calibration_steps=30)
    res_dyn = KalmanTAM(**kwargs).predict_online(dummy_panel_data)
    res_dyn = res_dyn[res_dyn["timestamp"] == dates[-1]].set_index("smart_meter_id")["KalmanAdapted_load"]

    model_stat = KalmanTAM(**kwargs)
    model_stat.fit(history)
    res_stat = model_stat.predict(last_day.drop(columns=["load"])).set_index("smart_meter_id")["KalmanAdapted_load"]

    np.testing.assert_allclose(
        res_stat.reindex(res_dyn.index).to_numpy(), res_dyn.to_numpy(), rtol=1e-9,
        err_msg="KalmanTAM fit() then predict() of the next day diverged from the online simulation."
    )
