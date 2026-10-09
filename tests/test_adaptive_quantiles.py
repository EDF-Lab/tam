# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
``AdaptiveTAM.predict_quantiles``: online quantiles of a distributional base, whose scale is rescaled by the standardized residuals of
the adaptive location, using past residuals only.
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.model import adaptative

N, BREAK, BASE_ROWS, WINDOW, UPDATE = 2000, 600, 500, 60, 10
_DIST = {"log_target": False, "tail_family": "normal"}


def _data(seed=0, n=N, break_at=BREAK, sd_after=3.0):
    """y = 2 + 1.5 x + noise, with a noise standard deviation of 1 that becomes ``sd_after`` at ``break_at``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    sd = np.where(np.arange(n) < break_at, 1.0, sd_after)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "x": x, "y": 2.0 + 1.5 * x + sd * rng.normal(size=n)})


def _base(df):
    base = ta.StaticTAM({"mu": "y ~ l(x)", "sigma": "y ~ l(x)"}, loss={"mu": "l2", "sigma": "gamma"}, date_col="date", dist_kwargs=_DIST)
    return base.fit(df.iloc[:BASE_ROWS])


def _adaptive(df, horizon=1):
    return ta.AdaptiveTAM(base_model=_base(df), adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                          training_window_periods=WINDOW, steps_per_period=1, horizon_steps=horizon, date_col="date", default_alpha_p=-6.0)


def test_a_plain_base_has_no_quantiles():
    df = _data()
    plain = ta.StaticTAM("y ~ l(x)", date_col="date").fit(df.iloc[:BASE_ROWS])
    model = ta.AdaptiveTAM(base_model=plain, adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                           training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    with pytest.raises(ValueError, match="distributional"):
        model.predict_quantiles(df, (0.1, 0.9))


def test_levels_outside_zero_one_are_rejected():
    df = _data()
    with pytest.raises(ValueError, match="taus"):
        _adaptive(df).predict_quantiles(df, (0.0, 0.5))


def test_the_interval_follows_a_variance_break():
    df = _data()
    quantiles = _adaptive(df).predict_quantiles(df, (0.1, 0.5, 0.9))
    late = (df["date"] >= df["date"].iloc[BREAK + 2 * WINDOW]).to_numpy()
    y = df["y"].to_numpy()
    inside = (y >= quantiles["q0.1"].to_numpy()) & (y <= quantiles["q0.9"].to_numpy())
    assert abs(float(inside[late].mean()) - 0.8) < 0.03


def test_without_the_online_scale_the_interval_is_too_narrow_after_the_break(monkeypatch):
    """The base alone (s = 1) learned a noise standard deviation of 1: after the break its 80% interval covers about a third."""
    monkeypatch.setattr(adaptative, "_online_scale", lambda standardized, *args: np.ones(len(standardized)))
    df = _data()
    quantiles = _adaptive(df).predict_quantiles(df, (0.1, 0.9))
    late = (df["date"] >= df["date"].iloc[BREAK + 2 * WINDOW]).to_numpy()
    y = df["y"].to_numpy()
    inside = (y >= quantiles["q0.1"].to_numpy()) & (y <= quantiles["q0.9"].to_numpy())
    assert float(inside[late].mean()) < 0.5


def test_quantiles_do_not_decrease_with_the_level():
    df = _data()
    quantiles = _adaptive(df).predict_quantiles(df, (0.05, 0.25, 0.5, 0.75, 0.95))
    values = quantiles.to_numpy()
    assert (np.diff(values, axis=1) >= 0).all() and np.isfinite(values).all()
    assert list(quantiles.columns) == ["q0.05", "q0.25", "q0.5", "q0.75", "q0.95"]


@pytest.mark.parametrize("horizon", [1, 3])
def test_no_look_ahead_a_later_target_does_not_move_an_earlier_quantile(horizon):
    df = _data()
    before = _adaptive(df, horizon).predict_quantiles(df, (0.1, 0.9))
    changed = df.copy()
    late = np.arange(len(df)) >= 1200
    changed.loc[late, "y"] = np.random.default_rng(1).normal(size=int(late.sum())) * 50.0
    after = _adaptive(df, horizon).predict_quantiles(changed, (0.1, 0.9))
    np.testing.assert_allclose(before.to_numpy()[:1200], after.to_numpy()[:1200], rtol=0, atol=1e-9)
    assert not np.allclose(before.to_numpy()[1300:], after.to_numpy()[1300:])


def test_with_a_scale_of_one_the_quantiles_are_the_base_sigma_around_the_adapted_location(monkeypatch):
    monkeypatch.setattr(adaptative, "_online_scale", lambda standardized, *args: np.ones(len(standardized)))
    df = _data()
    model = _adaptive(df)
    quantiles = model.predict_quantiles(df, (0.1, 0.5, 0.9))
    frame = model.predictions_
    location = frame["AdaptedEstimated__mu__"].to_numpy()
    _, sigma = model.distributional_base_._mu_sigma(frame)
    np.testing.assert_allclose(quantiles["q0.5"].to_numpy(), location, atol=1e-9)
    z = 1.2815515655446004
    np.testing.assert_allclose(quantiles["q0.9"].to_numpy() - location, sigma * z, rtol=1e-9)
    np.testing.assert_allclose(location - quantiles["q0.1"].to_numpy(), sigma * z, rtol=1e-9)


def test_online_scale_uses_only_residuals_known_at_the_origin():
    # window 10, interval 5, horizon 2: the first origin is row 11, then 16, 21, ...; an origin o reads the rows o-10 ... o-2
    residual = np.ones(40)
    residual[15] = 100.0                      # known from the origin 17 on, so first read at the origin 21
    scale = adaptative._online_scale(residual, np.zeros(40), np.arange(40), 10, 5, 2)
    assert (scale[:11] == 1.0).all()           # before the first origin: the base scale
    assert (scale[11:21] == 1.0).all()         # the spike at row 15 is not yet readable at the origins 11 and 16
    assert (scale[21:26] > 10.0).all()


def test_online_scale_keeps_the_base_scale_with_too_few_residuals():
    residual = np.full(30, np.nan)
    residual[:3] = 2.0
    scale = adaptative._online_scale(residual, np.zeros(30), np.arange(30), 10, 5, 1)
    assert (scale == 1.0).all()


def test_online_scale_works_group_by_group():
    residual = np.concatenate([np.full(30, 1.0), np.full(30, 3.0)])
    group = np.repeat([0, 1], 30)
    scale = adaptative._online_scale(residual, group, np.tile(np.arange(30), 2), 10, 5, 1)
    assert np.allclose(scale[20:30], 1.0) and np.allclose(scale[50:60], 3.0)


def test_a_plain_adaptive_model_is_unchanged_by_the_distributional_option():
    df = _data()
    plain = ta.StaticTAM("y ~ l(x)", date_col="date").fit(df.iloc[:BASE_ROWS])
    model = ta.AdaptiveTAM(base_model=plain, adaptive_formula="Residualy ~ l(effect_x)", update_interval_periods=UPDATE,
                           training_window_periods=WINDOW, steps_per_period=1, date_col="date", default_alpha_p=-6.0)
    assert model.distributional_base_ is None
    assert "EstimatedResidualy" in model.predict_online(df).columns
