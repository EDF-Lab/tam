# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
KalmanTAM is causal: its scaling comes from a reference period, its state is updated at every step, and its tuning
scores a period the user chooses.

* feature normalisation and target scale come from the first ``calibration_steps`` rows of each group (burn-in) or from
  ``calibration_data``, never from the whole online period;
* with ``horizon_steps=1`` the default ``block_size`` is 1 (the exact sequential filter);
* ``tune_hyperparameters`` scores the steps given by ``calibration_steps`` / ``calibration_mask``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta
from tam.model.kalman import KalmanTAM

N = 120          # rows per group
R = 90           # last row (0-based) that stays unchanged
CAL = 30         # reference rows
FORMULA = "load ~ l(temperature)"


def _panel(seed=0, n=N, groups=("A", "B")):
    rng = np.random.default_rng(seed)
    frames = []
    for j, g in enumerate(groups):
        temp = rng.uniform(-5, 30, n)
        frames.append(pd.DataFrame({
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="D"), "grp": g, "temperature": temp,
            "load": 50 + 30 * j + 0.5 * temp + rng.normal(0, 2, n)}))
    return pd.concat(frames, ignore_index=True)


def _perturbed(df, after=R):
    """Same data up to row `after` of each group; the later rows get another range (feature and target)."""
    out = df.copy()
    late = out.groupby("grp").cumcount() > after
    out.loc[late, "load"] = out.loc[late, "load"] * 3.0 + 400.0
    out.loc[late, "temperature"] = out.loc[late, "temperature"] * 2.0 + 50.0
    return out


def _kalman(**kwargs):
    kwargs.setdefault("block_size", 1)
    kwargs.setdefault("calibration_steps", CAL)
    return KalmanTAM(kalman_formula=FORMULA, group_col="grp", date_col="timestamp", **kwargs)


def _forecast(model, df, col="KalmanAdapted_load"):
    out = model.predict_online(df)
    return out.assign(_row=out.groupby("grp").cumcount())[["grp", "_row", col]]


# ----------------------------------------------------------------------------- causal scaling
def test_no_leak_forecasts_up_to_r_do_not_depend_on_later_rows():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _forecast(_kalman(), df)
        b = _forecast(_kalman(), _perturbed(df))
    early = a["_row"] <= R
    np.testing.assert_allclose(a.loc[early, "KalmanAdapted_load"].to_numpy(),
                               b.loc[early, "KalmanAdapted_load"].to_numpy(), rtol=0, atol=1e-9)
    assert not np.allclose(a.loc[~early, "KalmanAdapted_load"], b.loc[~early, "KalmanAdapted_load"])  # the perturbation is real


def test_no_leak_with_a_fitted_base_model():
    df = _panel(seed=1)
    base = ta.StaticTAM(formula="load ~ s(temperature, k=5)", group_col="grp", date_col="timestamp").fit(df[df.groupby("grp").cumcount() < 40])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _forecast(_kalman(base_model=base, use_decomposition=False), df)
        b = _forecast(_kalman(base_model=base, use_decomposition=False), _perturbed(df))
    early = a["_row"] <= R
    np.testing.assert_allclose(a.loc[early, "KalmanAdapted_load"].to_numpy(),
                               b.loc[early, "KalmanAdapted_load"].to_numpy(), rtol=0, atol=1e-9)


def test_constant_range_gives_the_whole_period_scaling():
    """When the reference rows already hold the extremes, the causal scaling equals the whole-period scaling of v1.3.x."""
    df = _panel(seed=2)
    first = df.groupby("grp").cumcount() < 2
    df.loc[first & (df.groupby("grp").cumcount() == 0), ["temperature", "load"]] = [-20.0, 0.0]
    df.loc[first & (df.groupby("grp").cumcount() == 1), ["temperature", "load"]] = [60.0, 300.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short = _forecast(_kalman(calibration_steps=5), df)
        long = _forecast(_kalman(calibration_steps=N - 1), df)
    np.testing.assert_allclose(short["KalmanAdapted_load"].to_numpy(), long["KalmanAdapted_load"].to_numpy(), rtol=0, atol=1e-10)


def test_calibration_data_scales_like_the_same_rows_in_front_of_the_data():
    df = _panel(seed=3)
    history, online = df[df.groupby("grp").cumcount() < 40], df[df.groupby("grp").cumcount() >= 40]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a separate calibration period has no burn-in, so no message
        out = _kalman(calibration_data=history, calibration_steps=None).predict_online(online)
    assert not out["KalmanAdapted_load"].isna().any() and len(out) == len(online)


def test_default_burn_in_gives_a_message_and_computes_every_row():
    df = _panel()
    with pytest.warns(UserWarning, match="reference period"):
        out = _kalman(calibration_steps=None).predict_online(df)
    assert len(out) == len(df) and not out["KalmanAdapted_load"].isna().any()


def test_default_calibration_steps_adapts_to_short_data_without_failing(dummy_panel_data):
    model = KalmanTAM(kalman_formula=FORMULA.replace("load", "load"), group_col="smart_meter_id", date_col="timestamp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.predict_online(dummy_panel_data)   # 100 rows per group, default 365 is capped at half the rows
    assert len(out) == len(dummy_panel_data)
    assert model.calibration_steps_used_ == 50


def test_calibration_steps_too_high_raises_a_clear_error():
    df = _panel(n=40)
    with pytest.raises(ValueError, match="calibration_steps"):
        _kalman(calibration_steps=40).predict_online(df)
    with pytest.raises(ValueError, match="calibration_steps"):
        _kalman(calibration_steps=500).predict_online(df)


def test_predict_reuses_the_scaling_stored_at_fit():
    df = _panel(seed=4)
    train, new = df[df.groupby("grp").cumcount() < 80], df[df.groupby("grp").cumcount() >= 80]
    model = _kalman()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train)
        stored = {g: v.clone() for g, v in model.scale_dict_.items()}
        norm = {g: {k: v for k, v in p.items()} for g, p in model.feature_extractor_.norm_params_.items()}
        a = model.predict(new.drop(columns=["load"]))
        model.predict(_perturbed(new).drop(columns=["load"]))
    assert all(torch.equal(model.scale_dict_[g], stored[g]) for g in stored)
    assert model.feature_extractor_.norm_params_.keys() == norm.keys()
    assert not a["KalmanAdapted_load"].isna().any()


# ----------------------------------------------------------------------------- update at every step
def test_default_block_size_is_one_and_the_forecast_follows_the_previous_step():
    model = KalmanTAM(kalman_formula=FORMULA, group_col="grp", date_col="timestamp", calibration_steps=CAL)
    assert model.block_size_ == 1
    df = _panel(seed=5)
    bumped = df.copy()
    t = 60                                                    # the step whose target is changed
    bumped.loc[bumped.groupby("grp").cumcount() == t, "load"] += 20.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _forecast(model, df)
        b = _forecast(KalmanTAM(kalman_formula=FORMULA, group_col="grp", date_col="timestamp", calibration_steps=CAL), bumped)
    col = "KalmanAdapted_load"
    at_t = a[a["_row"] == t][col].to_numpy(), b[b["_row"] == t][col].to_numpy()
    next_step = a[a["_row"] == t + 1][col].to_numpy(), b[b["_row"] == t + 1][col].to_numpy()
    np.testing.assert_allclose(*at_t, atol=1e-9)              # a forecast never sees its own target
    assert np.all(np.abs(next_step[0] - next_step[1]) > 1e-6)  # the next step does (v1.3.1: 128-step blocks, no update)


def test_explicit_block_size_above_one_warns_once_and_is_kept():
    with pytest.warns(UserWarning, match="every 16 steps"):
        model = KalmanTAM(kalman_formula=FORMULA, group_col="grp", date_col="timestamp", block_size=16)
    assert model.block_size_ == 16


def test_block_size_one_equals_a_plain_numpy_kalman_filter():
    df = _panel(seed=6)
    model = _kalman(process_noise_var=1e-3, observation_noise_var=0.7, offset_boost=10.0, P_init_diag=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = model.prepare_data(df)
        got = model._run_filter(prepared, 2.0, 0.7, 1e-3)
    phi = prepared["phi_matrix"].cpu().numpy().astype(float)
    y = prepared["y_stacked"].cpu().numpy().astype(float)
    base = prepared["base_pred_stacked"].cpu().numpy().astype(float)
    scale, center = prepared["y_scale"].cpu().numpy(), prepared["y_center"].cpu().numpy()
    for g in range(phi.shape[0]):
        d = phi.shape[-1]
        theta, P = np.zeros(d), np.eye(d) * 2.0
        Q = np.eye(d) * 1e-3
        P[0, 0] *= 10.0
        Q[0, 0] *= 10.0
        expected = np.zeros(phi.shape[1])
        for t in range(phi.shape[1]):
            x = phi[g, t]
            expected[t] = base[g, t, 0] + x @ theta
            s = 0.7 + x @ P @ x + 1e-6
            k = P @ x / s
            theta = theta + k * (y[g, t, 0] - expected[t])
            P = P - np.outer(k, x @ P) + Q
            P = (P + P.T) / 2
        np.testing.assert_allclose(got[g, :, 0].cpu().numpy(), expected * scale[g, 0, 0] + center[g, 0, 0], rtol=0, atol=1e-10)


# ----------------------------------------------------------------------------- tuning without look-ahead
GRID = {"observation_noise_var": [0.5, 1.0], "process_noise_var": [1e-4, 1e-2]}


def _tune(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _kalman().tune_hyperparameters(df, GRID, **kwargs)


def test_tuning_does_not_depend_on_rows_after_the_calibration_period():
    df = _panel(seed=7)
    window = slice(40, 80)
    params_a, rmse_a = _tune(df, calibration_steps=window)
    params_b, rmse_b = _tune(_perturbed(df, after=80), calibration_steps=window)
    assert params_a == params_b
    assert rmse_a == pytest.approx(rmse_b, abs=1e-9)


def test_tuning_accepts_a_slice_or_the_equivalent_mask():
    df = _panel(seed=8)
    mask = np.zeros(N, dtype=bool)
    mask[40:80] = True
    by_slice = _tune(df, calibration_steps=slice(40, 80))
    by_mask = _tune(df, calibration_mask=mask)
    assert by_slice[0] == by_mask[0] and by_slice[1] == pytest.approx(by_mask[1])


def _warnings_of(fn):
    """Runs `fn` and returns (result, [(category, message)]) of every warning it issued: nothing leaks into the test report."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [(w.category, str(w.message)) for w in caught]


def test_lookback_days_warns_and_keeps_the_old_window():
    df = _panel(seed=9)
    old, issued = _warnings_of(lambda: _kalman().tune_hyperparameters(df, GRID, lookback_days=1))
    assert any(c is FutureWarning and "lookback_days" in m for c, m in issued)
    # old rule: the last lookback_days * 24 steps of the data
    explicit = _tune(df, calibration_steps=slice(N - 1 - 24, N))
    assert old[0] == explicit[0] and old[1] == pytest.approx(explicit[1])


def test_tuning_without_a_period_warns_about_the_legacy_window():
    df = _panel(seed=10)
    _, issued = _warnings_of(lambda: _kalman().tune_hyperparameters(df, GRID))
    assert any(c is FutureWarning and "calibration_steps" in m for c, m in issued)
    assert any("reference period" in m for _, m in issued)           # the burn-in message comes with it


# ----------------------------------------------------------------------------- the operational loop equals the simulation
def _next_day_matches_the_simulation(df, make, steps=(41, 60, 90)):
    sim = make().predict_online(df)
    sim = sim.assign(_row=sim.groupby("grp").cumcount())
    for t in steps:
        history = df[df.groupby("grp").cumcount() < t]
        today = df[df.groupby("grp").cumcount() == t]
        model = make().fit(history)
        pred = model.predict(today.drop(columns=["load"]))
        expected = sim[sim["_row"] == t].set_index("grp")["KalmanAdapted_load"]
        got = pred.set_index("grp")["KalmanAdapted_load"]
        np.testing.assert_allclose(got.reindex(expected.index).to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9,
                                   err_msg=f"fit(history of {t} days).predict(day {t}) differs from the simulation")


def test_fit_then_predict_the_next_day_equals_the_online_simulation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _next_day_matches_the_simulation(_panel(seed=11), lambda: _kalman(process_noise_var=1e-3))


def test_fit_then_predict_the_next_day_equals_the_simulation_with_a_base_model_and_calibration_data():
    df = _panel(seed=12)
    base = ta.StaticTAM(formula="load ~ s(temperature, k=5)", group_col="grp", date_col="timestamp").fit(df[df.groupby("grp").cumcount() < 40])
    reference = df[df.groupby("grp").cumcount() < 40]
    online = df[df.groupby("grp").cumcount() >= 40].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = _kalman(base_model=base, use_decomposition=False, calibration_data=reference, calibration_steps=None).predict_online(online)
        sim = sim.assign(_row=sim.groupby("grp").cumcount())
        for t in (1, 5, 30):
            history = online[online.groupby("grp").cumcount() < t]
            today = online[online.groupby("grp").cumcount() == t]
            model = _kalman(base_model=base, use_decomposition=False, calibration_data=reference, calibration_steps=None).fit(history)
            got = model.predict(today.drop(columns=["load"])).set_index("grp")["KalmanAdapted_load"]
            expected = sim[sim["_row"] == t].set_index("grp")["KalmanAdapted_load"]
            np.testing.assert_allclose(got.reindex(expected.index).to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)
