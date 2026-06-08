r"""
Unit tests for ``tam.model.kalman.KalmanTAM``, the Block Kalman Filter
meta-learner that tracks drifting coefficients online.
"""

import warnings
import numpy as np
import pytest

import tam as ta
from tam.model.kalman import KalmanTAM


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
