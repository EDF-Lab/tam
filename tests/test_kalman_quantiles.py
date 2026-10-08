# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``KalmanTAM.predict_quantiles``: Gaussian quantiles from the predictive variance of the filter, ``x' P x + R`` in the target scale,
for the one-step forecast and for the delayed state of a multi-step horizon.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import tam as ta

N, SIGMA = 4000, 2.0
CALIBRATION = 365


def _frame(seed=0, n=N, groups=1):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(groups):
        x = rng.normal(size=n)
        frames.append(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "g": g, "x": x,
                                    "y": 5.0 + 2.0 * x + SIGMA * rng.normal(size=n)}))
    return pd.concat(frames, ignore_index=True)


def _model(df, observation_noise_var=None, horizon=1, groups=1):
    """A tracker with a nearly constant coefficient; R is the true noise variance in the standardised target space."""
    kwargs = dict(kalman_formula="y ~ l(x)", group_col="g" if groups > 1 else None, date_col="date", horizon_steps=horizon,
                  process_noise_var=1e-8, calibration_steps=CALIBRATION)
    if observation_noise_var is None:
        probe = ta.KalmanTAM(**kwargs)
        probe.prepare_data(df)
        scale = float(next(iter(probe._reference_scale_.values())))
        observation_noise_var = SIGMA ** 2 / scale ** 2
    return ta.KalmanTAM(observation_noise_var=observation_noise_var, **kwargs)


def test_the_interval_covers_when_the_noise_is_known():
    df = _frame()
    quantiles = _model(df).predict_quantiles(df, (0.1, 0.5, 0.9))
    y = df["y"].to_numpy()[CALIBRATION:]
    inside = (y >= quantiles["q0.1"].to_numpy()[CALIBRATION:]) & (y <= quantiles["q0.9"].to_numpy()[CALIBRATION:])
    assert abs(float(inside.mean()) - 0.8) < 0.03


def test_the_median_is_the_online_forecast():
    df = _frame()
    model = _model(df)
    quantiles = model.predict_quantiles(df, (0.1, 0.5, 0.9))
    forecast = model.predict_online(df)["KalmanAdapted_y"].to_numpy()
    np.testing.assert_allclose(quantiles["q0.5"].to_numpy(), forecast, rtol=0, atol=1e-9)


def test_quantiles_do_not_decrease_with_the_level_and_the_columns_are_named_by_level():
    df = _frame(n=1000)
    quantiles = _model(df).predict_quantiles(df, (0.05, 0.25, 0.5, 0.75, 0.95))
    assert list(quantiles.columns) == ["q0.05", "q0.25", "q0.5", "q0.75", "q0.95"]
    assert (np.diff(quantiles.to_numpy(), axis=1) > 0).all()


def test_levels_outside_zero_one_are_rejected():
    df = _frame(n=600)
    with pytest.raises(ValueError, match="taus"):
        _model(df).predict_quantiles(df, (0.5, 1.0))


def test_no_look_ahead_a_later_target_does_not_move_an_earlier_quantile():
    df = _frame()
    model = _model(df)
    before = model.predict_quantiles(df, (0.1, 0.9)).to_numpy()
    changed = df.copy()
    changed.loc[2000:, "y"] = np.random.default_rng(1).normal(size=len(df) - 2000) * 40.0
    after = _model(df).predict_quantiles(changed, (0.1, 0.9)).to_numpy()
    np.testing.assert_allclose(before[:2000], after[:2000], rtol=0, atol=1e-9)
    assert not np.allclose(before[2100:], after[2100:])


def test_the_predictive_variance_does_not_depend_on_the_target():
    """The covariance of a linear Gaussian filter is a function of the design and the noise, not of the observations."""
    df = _frame(n=1500)
    model = _model(df)
    shifted = df.assign(y=df["y"] * 3.0 + 7.0)
    other = _model(shifted, observation_noise_var=model.observation_noise_var_)
    width = lambda m, frame: (lambda q: q["q0.9"] - q["q0.1"])(m.predict_quantiles(frame, (0.1, 0.9))).to_numpy()
    # the target scale is 3 times larger, the standardised variance is the same: the widths differ by that factor only
    np.testing.assert_allclose(width(other, shifted) / width(model, df), 3.0, rtol=1e-6)


def test_a_longer_horizon_gives_wider_intervals():
    df = _frame(n=1500)
    one = _model(df, horizon=1).predict_quantiles(df, (0.1, 0.9))
    four = _model(df, horizon=4).predict_quantiles(df, (0.1, 0.9))
    width_one = (one["q0.9"] - one["q0.1"]).to_numpy()[CALIBRATION:]
    width_four = (four["q0.9"] - four["q0.1"]).to_numpy()[CALIBRATION:]
    assert np.isfinite(width_four).all() and (width_four >= width_one - 1e-12).all()


def test_the_groups_are_scored_separately():
    df = _frame(n=800, groups=2)
    quantiles = _model(df, groups=2).predict_quantiles(df, (0.1, 0.9))
    assert len(quantiles) == len(df)
    assert np.isfinite(quantiles.to_numpy()).all()


def test_asking_for_the_variance_does_not_change_the_filter():
    df = _frame(n=1000)
    model = _model(df)
    prepared = model.prepare_data(df)
    plain = model._run_filter(prepared, model.P_init_diag_, model.observation_noise_var_, model.process_noise_var_).clone()
    with_variance = model._run_filter(prepared, model.P_init_diag_, model.observation_noise_var_, model.process_noise_var_,
                                      with_variance=True)
    assert torch.equal(plain, with_variance)
