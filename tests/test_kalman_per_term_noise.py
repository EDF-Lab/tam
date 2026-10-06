# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
``KalmanTAM(process_noise_var={term: q, "offset": q, "default": q})``: one process noise per formula term (a diagonal ``Q``), every design column of a term getting the term's ``q``. A float keeps today's behaviour, bit for bit.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tam.model.kalman import KalmanTAM

FORMULA = "load ~ l(temperature) + l(wind)"
N = 160
JUMP = 80


def _panel(seed=0, n=N, jump=True):
    """One group; the load follows 0.5 * temperature + b * wind with b = 1 and, after JUMP, b = 4."""
    rng = np.random.default_rng(seed)
    temp, wind = rng.uniform(-5, 30, n), rng.uniform(0, 10, n)
    b = np.where(np.arange(n) < JUMP, 1.0, 4.0) if jump else 1.0
    return pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=n, freq="D"), "temperature": temp, "wind": wind,
                         "load": 50 + 0.5 * temp + b * wind + rng.normal(0, 0.5, n)})


def _kalman(noise=1e-4, **kwargs):
    kwargs.setdefault("block_size", 1)
    kwargs.setdefault("calibration_steps", 30)
    return KalmanTAM(kalman_formula=FORMULA, date_col="timestamp", process_noise_var=noise, **kwargs)


def _forecast(noise=1e-4, **kwargs):
    model = _kalman(noise, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.predict_online(_panel())
    return model, out["KalmanAdapted_load"].to_numpy()


# ------------------------------------------------------------------ the same q everywhere = the float call, bit for bit
@pytest.mark.parametrize("q", [1e-4, 1e-2, 0.0])
def test_a_default_only_dict_equals_the_float_call(q):
    _, reference = _forecast(q)
    _, from_dict = _forecast({"default": q})
    np.testing.assert_array_equal(from_dict, reference)


def test_every_term_named_with_the_same_q_equals_the_float_call():
    _, reference = _forecast(1e-3)
    _, from_dict = _forecast({"default": 1e-3, "temperature": 1e-3, "wind": 1e-3})   # the offset is not in the dict: default * its boost
    np.testing.assert_array_equal(from_dict, reference)


def test_a_float_that_is_an_int_or_numpy_scalar_is_still_a_float():
    _, reference = _forecast(1e-3)
    np.testing.assert_array_equal(_forecast(np.float64(1e-3))[1], reference)


# ------------------------------------------------------------------ one q per term changes only that term
def test_a_different_q_for_one_term_changes_the_forecast():
    _, reference = _forecast(1e-4)
    _, other = _forecast({"default": 1e-4, "wind": 1e-2})
    assert not np.allclose(other, reference)


def test_a_term_with_q_zero_follows_a_changing_coefficient_much_less():
    """After a jump of the true wind coefficient, a free wind term follows it; with q = 0 its variance shrinks and it follows far less."""
    free, _ = _forecast({"default": 1e-4, "wind": 1e-1})
    frozen, _ = _forecast({"default": 1e-4, "wind": 0.0})
    wind = 3                                             # columns: systematic offset, intercept, temperature, wind
    path_free, path_frozen = free.states_history_[:, 0, wind].numpy(), frozen.states_history_[:, 0, wind].numpy()
    moved_free = abs(path_free[-1] - path_free[JUMP - 1])
    moved_frozen = abs(path_frozen[-1] - path_frozen[JUMP - 1])
    assert moved_free > 2.0 * moved_frozen, (moved_free, moved_frozen)


def test_the_offset_key_removes_the_offset_boost_on_its_noise():
    _, boosted = _forecast({"default": 1e-4})                        # offset q = 1e-4 * offset_boost
    _, explicit = _forecast({"default": 1e-4, "offset": 1e-4})      # offset q = 1e-4: no boost
    assert not np.allclose(boosted, explicit)


# ------------------------------------------------------------------ term names
def test_term_columns_names_the_terms_and_counts_their_design_columns():
    model = _kalman()
    assert model.term_columns() == {"offset": 2, "temperature": 1, "wind": 1}      # offset: the systematic column and the intercept effect


def test_an_unknown_term_name_is_rejected_with_the_valid_ones():
    with pytest.raises(ValueError, match="temperature") as excinfo:
        _forecast({"default": 1e-4, "tempratur": 1e-3})
    assert "tempratur" in str(excinfo.value) and "wind" in str(excinfo.value)


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf")])
def test_a_negative_or_non_finite_q_is_rejected(bad):
    with pytest.raises(ValueError, match="process_noise_var"):
        _forecast({"default": 1e-4, "wind": bad})
    with pytest.raises(ValueError, match="process_noise_var"):
        _forecast(bad)


# ------------------------------------------------------------------ tuning accepts per-term candidates
def test_tune_hyperparameters_scores_per_term_candidates():
    model = _kalman()
    grid = {"observation_noise_var": [1.0], "process_noise_var": [1e-4, {"default": 1e-4, "wind": 1e-2}, {"default": 1e-4, "wind": 0.0}]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best, score = model.tune_hyperparameters(_panel(), grid, calibration_steps=slice(100, 160))
    assert best["process_noise_var"] in grid["process_noise_var"] and np.isfinite(score)
    assert best["process_noise_var"] == {"default": 1e-4, "wind": 1e-2}        # the coefficient jumps: the free wind term wins
