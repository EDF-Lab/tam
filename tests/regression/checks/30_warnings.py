# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The warnings a user sees: how many, which class, and which part of the message, in the situations tam warns about (and the ones it must stay quiet in)."""
import warnings

import tam as ta
from common import BASE, DATE, GROUP, frame

SPLINE = "load ~ s(temperature, k=8) + " + BASE
LINEAR = "load ~ l(temperature) + " + BASE


def run(res):
    train, test = res.train, res.test
    hotter = test.assign(temperature=test["temperature"] + 25.0)

    def observed(name, call):
        def step():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                call()
            ours = [w for w in caught if isinstance(w.message, ta.TAMExtrapolationWarning) or str(w.message).startswith("TAM")]
            res.value(f"{name}.count", len(ours))
            res.value(f"{name}.classes", ", ".join(sorted({type(w.message).__name__ for w in ours})) or "none")
            res.value(f"{name}.first", (str(ours[0].message)[:70] if ours else "none"))

        res.attempt(f"{name}.harness", step)

    def fitted(formula, **options):
        return ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE, **options).fit(train)

    def twice():
        model = fitted(SPLINE)
        model.predict(hotter)
        with warnings.catch_warnings(record=True) as again:         # the warning is raised once per model: the second forecast is silent
            warnings.simplefilter("always")
            model.predict(hotter)
        res.value("spline_beyond_twice.second_forecast_warnings", len([w for w in again if isinstance(w.message, ta.TAMExtrapolationWarning)]))

    observed("spline_inside_the_range", lambda: fitted(SPLINE).predict(test))
    observed("spline_beyond_the_range", lambda: fitted(SPLINE).predict(hotter))
    res.attempt("spline_beyond_twice", twice)
    observed("linear_beyond_the_range", lambda: fitted(LINEAR).predict(hotter))
    observed("constant_extrapolation", lambda: fitted("load ~ s(temperature, k=8, extrapolate='constant') + " + BASE).predict(hotter))
    weekdays = train[train["day_type_week"] < 5]
    observed("unseen_categorical_level", lambda: ta.StaticTAM(formula="load ~ s(temperature, k=8) + c(day_type_week, n_cat=7, topo='nominal') + l(load_d1)", group_col=GROUP,
                                                              date_col=DATE).fit(weekdays).predict(test))
    observed("missing_expected_levels_at_fit", lambda: ta.StaticTAM(formula="load ~ l(temperature) + c(day_type_week, n_cat=7, topo='nominal')", group_col=GROUP,
                                                                    date_col=DATE).fit(weekdays))
    observed("adaptive_in_range", lambda: ta.AdaptiveTAM(adaptive_formula="load ~ s(temperature, k=6) + l(load_d1)", update_interval_periods=7,
                                                         training_window_periods=56, steps_per_period=1, group_col=GROUP, date_col=DATE).predict_online(frame()))

    def rare_level():
        df = res.train.sort_values(DATE).reset_index(drop=True)
        df["event"] = 0
        df.loc[df[DATE] >= df[DATE].max() - (df[DATE].max() - df[DATE].min()) / 10, "event"] = 1
        ta.AdaptiveTAM(adaptive_formula="load ~ l(load_d1) + c(event, n_cat=2, topo='nominal')", update_interval_periods=7, training_window_periods=40,
                       steps_per_period=1, group_col=GROUP, date_col=DATE).predict_online(df)

    observed("adaptive_level_absent_from_its_window", rare_level)
