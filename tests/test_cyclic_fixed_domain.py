# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
A Fourier term can declare the period of its input: ``f(toy, cyclic=True, period=(0, 1))``.

Every feature is normalised to [-1, 1] with the minimum and maximum of its training rows, and a cyclic term takes [-1, 1] as one
period. With a time of year in [0, 1] and a history from January to June, the training range [0, 0.5] became one whole period:
January 1st and June 30th were the same point of the cycle and the forecast of July started the year again. A declared period is
the range the feature is normalised on instead. Without ``period`` nothing changes (the training range is the period, as before).
"""
import numpy as np
import pandas as pd
import pytest

import tam as ta
from tam.common.utils import parse_formula_to_terms
from tam.model._data import _fit_normalization_params
from tam.model.spectrum import fixed_ranges
from tam.model.spectrum._factory import create_effects_from_parsed_terms

AMPLITUDE = 3.0


def _daily_yearly_sine(days: int, start: str = "2023-01-01", noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    toy = (dates.dayofyear.to_numpy() - 1) / 365.0                       # time of year on [0, 1), the usual coding
    rng = np.random.default_rng(seed)
    y = 10.0 + AMPLITUDE * np.sin(2.0 * np.pi * toy) + noise * rng.standard_normal(days)
    return pd.DataFrame({"date": dates, "toy": toy, "y": y})


def _half_year_error(formula: str) -> float:
    data = _daily_yearly_sine(365)
    train, test = data.iloc[:182], data.iloc[182:]                       # January to June, then July to December
    model = ta.StaticTAM(formula, date_col="date").fit(train)
    return float(np.abs(model.predict(test)["Estimatedy"].to_numpy() - test["y"].to_numpy()).max())


def test_declared_period_reads_half_a_year_as_half_a_cycle():
    assert _half_year_error("y ~ f(toy, m=1, s=1, cyclic=True, period=(0, 1))") <= 0.1 * AMPLITUDE


def test_without_period_half_a_year_is_taken_as_the_whole_cycle():
    # The default is unchanged: the training range is the period, which is wrong for a partial cycle (documented behaviour).
    assert _half_year_error("y ~ f(toy, m=1, s=1, cyclic=True)") > AMPLITUDE


def test_values_outside_the_period_wrap_around():
    data = _daily_yearly_sine(365 * 2, noise=0.1)
    model = ta.StaticTAM("y ~ f(toy, m=3, s=1, cyclic=True, period=(0, 1))", date_col="date").fit(data)
    probe = data.iloc[:2].assign(toy=[0.02, 1.02])                       # 1.02 is early January of the next cycle
    inside, outside = model.predict(probe)["Estimatedy"].to_numpy()
    assert inside == pytest.approx(outside, abs=1e-9)


def test_adaptive_windows_shorter_than_the_cycle_use_the_declared_period():
    # Each 60-day window was normalised on its own rows: two months were read as a whole year, even with years of data.
    data = _daily_yearly_sine(365 * 2)
    data["grp"] = "a"
    model = ta.AdaptiveTAM(adaptive_formula="y ~ f(toy, m=1, s=1, cyclic=True, period=(0, 1))", update_interval_periods=7,
                           training_window_periods=60, steps_per_period=1, group_col="grp", date_col="date")
    scored = model.predict_online(data).dropna(subset=["AdaptedEstimatedy"])
    error = np.abs(scored["AdaptedEstimatedy"].to_numpy() - scored["y"].to_numpy())
    assert np.median(error) <= 0.05 * AMPLITUDE


def _effects(formula: str):
    return create_effects_from_parsed_terms(parse_formula_to_terms(formula)[1], token_values={}, default_alpha_p=-9.0, data_info={})


def test_fixed_ranges_rules():
    assert fixed_ranges(_effects("y ~ f(toy, m=2, cyclic=True, period=(0, 1))")) == {"toy": (0.0, 1.0)}
    assert fixed_ranges(_effects("y ~ f(toy, m=2, cyclic=True)")) == {}                      # no period: min/max as before
    assert fixed_ranges(_effects("y ~ f(hour, m=2, cyclic=True, period=(0, 24)) + l(hour)")) == {"hour": (0.0, 24.0)}
    assert fixed_ranges(_effects("y ~ te(f(toy, m=2, cyclic=True, period=(0, 1)), s(x))")) == {"toy": (0.0, 1.0)}
    assert fixed_ranges(_effects("y ~ f(toy, m=2, cyclic=True, period=(0, 1)) + c(toy, n_cat=4)")) == {}
    with pytest.raises(ValueError, match="Two periods"):
        fixed_ranges(_effects("y ~ f(toy, m=2, period=(0, 1)) + f(toy, m=3, period=(0, 2))"))
    with pytest.raises(ValueError, match="period"):
        _effects("y ~ f(toy, m=2, period=(1, 0))")


def test_normalization_uses_the_declared_range():
    frame = pd.DataFrame({"g": 0, "toy": np.linspace(0.1, 0.4, 50), "x": np.linspace(5.0, 9.0, 50)})
    params, _ = _fit_normalization_params(frame, ["toy", "x"], "g", fixed_ranges={"toy": (0.0, 1.0)})
    assert params[0]["min"]["toy"] == 0.0 and params[0]["max"]["toy"] == 1.0
    assert params[0]["min"]["x"] == 5.0 and params[0]["max"]["x"] == 9.0                   # other features unchanged


def test_integer_hours_need_the_period_to_tell_23h_from_0h():
    # Hours 0..23 normalised on their own range put 23 on +1 and 0 on -1, the same point of a cyclic basis.
    hours = np.tile(np.arange(24), 200).astype(float)
    profile = np.where(hours == 23, 8.0, np.where(hours == 0, 2.0, 5.0 + np.sin(2 * np.pi * hours / 24)))
    data = pd.DataFrame({"hour": hours, "y": profile + 0.1 * np.random.default_rng(0).standard_normal(len(hours))})
    probe = pd.DataFrame({"hour": [0.0, 23.0], "y": [0.0, 0.0]})
    glued = ta.StaticTAM("y ~ f(hour, m=10, s=1, cyclic=True)").fit(data).predict(probe)["Estimatedy"].to_numpy()
    apart = ta.StaticTAM("y ~ f(hour, m=10, s=1, cyclic=True, period=(0, 24))").fit(data).predict(probe)["Estimatedy"].to_numpy()
    assert glued[0] == pytest.approx(glued[1], abs=1e-6)
    assert apart[1] - apart[0] > 3.0


# --- period="auto": (first value, last value + one step), the step read from the training rows ------------------------
def test_auto_period_tells_23h_from_0h():
    hours = np.tile(np.arange(24), 200).astype(float)
    profile = np.where(hours == 23, 8.0, np.where(hours == 0, 2.0, 5.0 + np.sin(2 * np.pi * hours / 24)))
    data = pd.DataFrame({"hour": hours, "y": profile + 0.1 * np.random.default_rng(0).standard_normal(len(hours))})
    probe = pd.DataFrame({"hour": [0.0, 23.0], "y": [0.0, 0.0]})
    auto = ta.StaticTAM("y ~ f(hour, m=10, s=1, cyclic=True, period='auto')").fit(data).predict(probe)["Estimatedy"].to_numpy()
    given = ta.StaticTAM("y ~ f(hour, m=10, s=1, cyclic=True, period=(0, 24))").fit(data).predict(probe)["Estimatedy"].to_numpy()
    np.testing.assert_allclose(auto, given, atol=1e-9)


@pytest.mark.parametrize("values, expected", [
    (np.arange(24.0), (0.0, 24.0)),                         # hours
    (np.arange(1.0, 13.0), (1.0, 13.0)),                    # months
    (np.arange(365) / 365.0, (0.0, 1.0)),                   # time of year, one value per day
])
def test_auto_period_is_first_value_to_last_plus_one_step(values, expected):
    from tam.model._data import _resolve_fixed_ranges
    frame = pd.DataFrame({"x": np.tile(values, 3)})
    low, high = _resolve_fixed_ranges(frame, {"x": "auto"})["x"]
    assert low == pytest.approx(expected[0]) and high == pytest.approx(expected[1])


def test_auto_period_in_fixed_ranges_and_conflicts():
    assert fixed_ranges(_effects("y ~ f(hour, m=2, cyclic=True, period='auto')")) == {"hour": "auto"}
    with pytest.raises(ValueError, match="Two periods"):
        fixed_ranges(_effects("y ~ f(hour, m=2, period='auto') + f(hour, m=3, period=(0, 24))"))
