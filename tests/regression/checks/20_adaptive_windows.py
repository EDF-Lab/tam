# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""AdaptiveTAM on its own: the training window, the update interval, the steps per period and the horizon of the sliding-window simulation.
The 28-day window with a spline is the deliberate stress case."""
import tam as ta
from common import DATE, GROUP, TARGET, frame, second_half

FORMULA = "load ~ l(load_d1) + l(temperature) + f(toy, m=4, s=1, cyclic=True) + c(day_type_week, n_cat=7, topo='nominal')"
SPLINE_STRESS = "load ~ l(load_d1) + s(temperature, k=6) + c(day_type_week, n_cat=7, topo='nominal')"


def simulate(window, interval, steps=1, horizon=1, formula=FORMULA):
    model = ta.AdaptiveTAM(adaptive_formula=formula, update_interval_periods=interval, training_window_periods=window, steps_per_period=steps,
                           horizon_steps=horizon, group_col=GROUP, date_col=DATE)
    return model.predict_online(frame())


def run(res):
    for window, interval in ((28, 1), (28, 7), (56, 7), (90, 7), (90, 30)):
        def step(window=window, interval=interval):
            out = simulate(window, interval)
            res.forecast(f"window{window}_every{interval}", *second_half(out, f"AdaptedEstimated{TARGET}" if f"AdaptedEstimated{TARGET}" in out else f"Estimated{TARGET}"))
            res.value(f"window{window}_every{interval}.warmup_rows_without_forecast", int(out[f"Estimated{TARGET}"].isna().sum()))

        res.attempt(f"window{window}_every{interval}", step)

    def stress():                                       # a spline on a 28-day window (28 rows per hour): the deliberate stress case
        for interval in (1, 7):
            out = simulate(28, interval, formula=SPLINE_STRESS)
            res.forecast(f"stress_spline_window28_every{interval}", *second_half(out, f"Estimated{TARGET}"))

    res.attempt("stress_spline_window28", stress)

    def horizon():
        out = simulate(56, 7, horizon=2)
        res.forecast("horizon2", *second_half(out, f"Estimated{TARGET}"))

    res.attempt("horizon2", horizon)

    def linear_only():
        out = simulate(56, 7, formula="load ~ l(load_d1) + l(load_d7) + l(temperature)")
        res.forecast("linear_only", *second_half(out, f"Estimated{TARGET}"))

    res.attempt("linear_only", linear_only)

    def final_state():
        model = ta.AdaptiveTAM(adaptive_formula=FORMULA, update_interval_periods=7, training_window_periods=56, steps_per_period=1, group_col=GROUP, date_col=DATE)
        model.fit(frame())
        res.forecast("fit_then_predict", frame()[TARGET].tail(500), model.predict(frame().tail(500))[f"Estimated{TARGET}"])

    res.attempt("fit_then_predict", final_state)
