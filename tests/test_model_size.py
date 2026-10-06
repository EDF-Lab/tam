# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Model size: a fitted model keeps what ``predict()`` needs, not caches that grow with the data.

* ``StaticTAM`` is small whatever the number of training rows.
* ``AdaptiveTAM`` frees its simulation cache (every overlapping window, stacked) once the run is done; ``prepare_simulation()`` alone still fills it.
* ``compact()`` strips what is left that grows with the data (``predictions_``, the state and weight histories, except their last row) and returns the
  model: its forecasts and its online continuation are unchanged to the last bit.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import tam as ta

pytestmark = pytest.mark.filterwarnings("ignore:KalmanTAM:UserWarning")


def _data(n=300, seed=0, groups=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    frames = []
    for g in groups:
        f = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "g": g, "x": rng.normal(size=n), "z": rng.normal(size=n)})
        f["y"] = f["x"] + 0.5 * f["z"] + rng.normal(0, 0.2, n)
        f["E1"] = f["y"] + rng.normal(0, 0.3, n)
        f["E2"] = f["y"] + rng.normal(0, 0.5, n)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _size(model):
    return len(pickle.dumps(model))


def _static(df):
    return ta.StaticTAM(formula="y ~ l(x) + s(z, k=8)", group_col="g", date_col="date").fit(df)


def _adaptive(df, online=True):
    model = ta.AdaptiveTAM(adaptive_formula="y ~ l(x) + s(z, k=6)", update_interval_periods=1, training_window_periods=60, steps_per_period=1,
                           group_col="g", date_col="date")
    return model.predict_online(df) if online else model, model


def _kalman(df):
    model = ta.KalmanTAM(kalman_formula="y ~ l(x) + l(z)", group_col="g", date_col="date", calibration_steps=60)
    model.predict_online(df)
    return model


def _opera(df):
    model = ta.OperaTAM("y ~ l(E1) + l(E2)", group_col="g", date_col="date")
    model.predict_online(df)
    return model


# ------------------------------------------------------------------ sizes
def test_a_static_model_does_not_grow_with_the_number_of_training_rows():
    short, long = _size(_static(_data(n=300))), _size(_static(_data(n=900)))
    assert abs(long - short) / short < 0.01, (short, long)


def test_adaptive_frees_its_simulation_cache_after_the_run():
    _, model = _adaptive(_data())
    assert model.simulation_data_ is None
    assert model.last_state_dict_ and model.predictions_ is not None


def test_adaptive_frees_its_simulation_cache_after_fit():
    df = _data()
    model = ta.AdaptiveTAM(adaptive_formula="y ~ l(x)", update_interval_periods=1, training_window_periods=60, steps_per_period=1,
                           group_col="g", date_col="date").fit(df)
    assert model.simulation_data_ is None and model.last_state_dict_


def test_prepare_simulation_alone_still_fills_the_cache():
    _, model = _adaptive(_data(), online=False)
    model.prepare_simulation(_data())
    assert model.simulation_data_ is not None


def test_a_run_pickles_much_smaller_than_the_simulation_cache():
    df = _data(n=600)
    _, cached = _adaptive(df, online=False)
    cached.prepare_simulation(df)
    _, run = _adaptive(df)
    assert _size(run) < 0.25 * _size(cached), (_size(run), _size(cached))


@pytest.mark.parametrize("build", [lambda df: _adaptive(df)[1], _kalman, _opera], ids=["adaptive", "kalman", "opera"])
def test_a_compact_model_does_not_grow_with_the_data(build):
    sizes = [_size(build(_data(n=n)).compact()) for n in (300, 900)]
    assert abs(sizes[1] - sizes[0]) / sizes[0] < 0.02, sizes


def test_compact_strips_the_histories_but_their_last_row():
    kalman, opera = _kalman(_data()), _opera(_data())
    full_k, full_o = kalman.states_history_.shape[0], {g: len(w) for g, w in opera.weights_history_.items()}
    before_k, before_o = _size(kalman), _size(opera)
    assert kalman.compact() is kalman and opera.compact() is opera
    assert kalman.states_history_.shape[0] == 1 < full_k
    assert all(len(w) == 1 for w in opera.weights_history_.values()) and all(n > 1 for n in full_o.values())
    assert _size(kalman) < before_k and _size(opera) < before_o


# ------------------------------------------------------------------ bit-identical forecasts and continuation
def test_compact_then_predict_is_bit_identical_for_the_four_models():
    df, future = _data(), _data(seed=5)
    static = _static(df)
    adaptive = _adaptive(df)[1]
    kalman, opera = _kalman(df), _opera(df)
    expected = {
        "static": static.predict(future)["Estimatedy"].to_numpy(),
        "adaptive": adaptive.predict(future)["AdaptedEstimatedy"].to_numpy(),
        "kalman": kalman.predict(future)["KalmanAdapted_y"].to_numpy(),
        "opera": opera.predict(future)["prediction_opera"].to_numpy(),
    }
    for model in (static, adaptive, kalman, opera):
        assert model.compact() is model
    np.testing.assert_array_equal(static.predict(future)["Estimatedy"].to_numpy(), expected["static"])
    np.testing.assert_array_equal(adaptive.predict(future)["AdaptedEstimatedy"].to_numpy(), expected["adaptive"])
    np.testing.assert_array_equal(kalman.predict(future)["KalmanAdapted_y"].to_numpy(), expected["kalman"])
    np.testing.assert_array_equal(opera.predict(future)["prediction_opera"].to_numpy(), expected["opera"])


def test_compact_keeps_the_state_that_continues_the_online_run():
    df = _data()
    adaptive, kalman = _adaptive(df)[1], _kalman(df)
    states = {g: s.clone() for g, s in adaptive.last_state_dict_.items()}
    kalman_states = {g: s.clone() for g, s in kalman.last_state_dict_.items()}
    adaptive.compact()
    kalman.compact()
    for g, s in states.items():
        assert (adaptive.last_state_dict_[g] == s).all()
    for g, s in kalman_states.items():
        assert (kalman.last_state_dict_[g] == s).all()


def test_compact_is_idempotent():
    df = _data()
    model = _opera(df)
    once = {g: w.copy() for g, w in model.compact().weights_history_.items()}
    twice = model.compact().weights_history_
    assert all((once[g] == twice[g]).all() for g in once)


# ------------------------------------------------------------------ what needs a history says so
def test_plotting_the_weights_after_compact_raises_a_clear_error():
    model = _opera(_data()).compact()
    with pytest.raises(RuntimeError, match="compact"):
        model.plot_weights(df=_data())


def test_grid_search_leaves_no_cache_on_the_model_it_searched_nor_on_the_result():
    df = _data(n=150, groups=("a", "b"))
    model = ta.AdaptiveTAM(adaptive_formula="y ~ s(x, k='gk')", update_interval_periods=1, training_window_periods=30, steps_per_period=1,
                           group_col="g", date_col="date")
    best = model.grid_search_fit(df, {"gk": [4, 6]})
    assert model.simulation_data_ is None and best.simulation_data_ is None and best.predictions_ is not None
