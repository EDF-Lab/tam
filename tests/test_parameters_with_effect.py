# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Which arguments of an effect change the forecast, and when.

The default penalty of a term is ``ap=-9`` (1e-9): next to it the weights of ``phys()``, the ``scaled`` factor of ``l()`` and the
difference order of an ordinal ``c()`` leave the forecast unchanged to the solver tolerance. With an explicit penalty (``ap=-2``)
they act. Each test checks both sides, so the documentation of the effect pages (``architecture/core/06_the_spectrum_api.md``)
stays true. ``f()`` has no ``scaled`` argument: a value given is ignored. ``extrapolate`` is only visible outside the training
range: a spline gives the same forecast with ``'continue'`` and ``'constant'``, a linear term the same with ``'continue'`` and
``'linear'``, a Fourier term does not.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import tam

N, N_TRAIN = 600, 450
HIDDEN = 1e-4     # largest forecast difference still called "unchanged" (the observed ones are about 1e-6)
ACTS = 1e-3       # smallest forecast difference called "acts" (the observed ones are about 1e-2 to 1)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, N)
    z = rng.uniform(-2, 2, N)
    c = rng.integers(0, 6, N)
    y = np.sin(x) + 0.5 * z + 0.3 * c + rng.normal(0, 0.3, N)
    return pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=N, freq="D"), "grp": "G",
                         "x": x, "z": z, "c": c, "y": y})


def _forecast(frame, formula, predict_on=None):
    model = tam.StaticTAM(formula=formula, group_col="grp", date_col="timestamp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame.iloc[:N_TRAIN])
        out = model.predict(frame if predict_on is None else predict_on)
    return out["Estimatedy"].to_numpy()[N_TRAIN:]


def _gap(frame, formula_a, formula_b):
    return float(np.abs(_forecast(frame, formula_a) - _forecast(frame, formula_b)).max())


def _beyond_the_range(frame):
    out = frame.copy()
    out.loc[N_TRAIN:, "x"] = out.loc[N_TRAIN:, "x"] * 1.6 + 2.0
    out.loc[N_TRAIN:, "z"] = out.loc[N_TRAIN:, "z"] * 2.0
    return out


@pytest.mark.parametrize("a, b", [("D1=1", "D2=1"), ("D2=1", "D1=1, D2=1"), ("D2=1", "D0=5, D2=1")])
def test_phys_weights_are_hidden_by_the_default_penalty_and_act_with_an_explicit_one(frame, a, b):
    assert _gap(frame, f"y ~ phys(x, {a})", f"y ~ phys(x, {b})") < HIDDEN
    assert _gap(frame, f"y ~ phys(x, {a}, ap=-2)", f"y ~ phys(x, {b}, ap=-2)") > ACTS


def test_linear_scaled_is_a_rescaling_hidden_by_the_default_penalty(frame):
    assert _gap(frame, "y ~ l(z)", "y ~ l(z, scaled=10)") < HIDDEN
    assert _gap(frame, "y ~ l(z, ap=-2)", "y ~ l(z, scaled=10, ap=-2)") > ACTS


def test_fourier_has_no_scaled_argument_a_value_given_is_ignored(frame):
    assert _gap(frame, "y ~ f(x)", "y ~ f(x, scaled=10)") == 0.0
    assert _gap(frame, "y ~ f(x, ap=-2)", "y ~ f(x, scaled=10, ap=-2)") == 0.0


def test_ordinal_difference_order_is_hidden_by_the_default_penalty(frame):
    assert _gap(frame, "y ~ c(c, topo='ordinal', p_order=1)", "y ~ c(c, topo='ordinal', p_order=2)") < HIDDEN
    assert _gap(frame, "y ~ c(c, topo='ordinal', p_order=1, ap=-2)", "y ~ c(c, topo='ordinal', p_order=2, ap=-2)") > ACTS


def _extrapolated(frame, term, mode):
    far = _beyond_the_range(frame)
    return _forecast(frame, f"y ~ {term.format(f', extrapolate={mode!r}')}", predict_on=far)


def test_a_spline_gives_the_same_forecast_with_continue_and_constant(frame):
    gap = np.abs(_extrapolated(frame, "s(x{})", "continue") - _extrapolated(frame, "s(x{})", "constant")).max()
    assert gap < 1e-9
    assert np.abs(_extrapolated(frame, "s(x{})", "continue") - _extrapolated(frame, "s(x{})", "linear")).max() > ACTS


def test_a_linear_term_gives_the_same_forecast_with_continue_and_linear_but_not_constant(frame):
    cont = _extrapolated(frame, "l(z{})", "continue")
    assert np.abs(cont - _extrapolated(frame, "l(z{})", "linear")).max() < 1e-9
    assert np.abs(cont - _extrapolated(frame, "l(z{})", "constant")).max() > ACTS


def test_a_fourier_term_continues_its_waves_beyond_the_range(frame):
    gap = np.abs(_extrapolated(frame, "f(x{})", "continue") - _extrapolated(frame, "f(x{})", "constant")).max()
    assert gap > ACTS


def test_pid_features_depend_on_the_frame_for_the_first_window_rows_only(frame):
    w = 7
    data = frame.copy()
    data["y_lag"] = data["y"].shift(1).bfill()
    model = tam.StaticTAM(formula=f"y ~ pid(y_lag, w={w}, ap=-2)", group_col="grp", date_col="timestamp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data.iloc[:N_TRAIN])
        long_frame = model.predict(data.iloc[N_TRAIN - 50:])["Estimatedy"].to_numpy()[50:]
        tail_only = model.predict(data.iloc[N_TRAIN:])["Estimatedy"].to_numpy()
        with_history = model.predict(data.iloc[N_TRAIN - w:])["Estimatedy"].to_numpy()[w:]
    gap = np.abs(tail_only - long_frame)
    assert gap[:w - 1].max() > ACTS       # the warm-up rows of a short frame are not the ones of the long frame
    assert gap[w:].max() < 1e-9           # from row w on the frame no longer matters
    assert np.abs(with_history - long_frame).max() < 1e-9   # w rows of history in front remove the difference
