# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hyper-parameter search by coordinate descent: a StaticTAM over knots and penalty, an AdaptiveTAM over knots."""
import pandas as pd

import tam as ta
from common import DATE, GROUP, TARGET, frame


def run(res):
    train, test = res.train, res.test
    validation = train[train[DATE] >= train[DATE].max() - pd.Timedelta(days=60)]
    fit = train[train[DATE] < train[DATE].max() - pd.Timedelta(days=60)]

    def static_search():
        model = ta.StaticTAM(formula="load ~ s(temperature, k='knots', ap='penalty') + l(load_d1) + c(day_type_week, n_cat=7, topo='nominal')",
                             group_col=GROUP, date_col=DATE)
        best = model.grid_search_fit(fit, validation, {"knots": [4, 8, 12], "penalty": [-8, -4, 0]})
        res.forecast("static", test[TARGET], best.predict(test)[f"Estimated{TARGET}"])

    res.attempt("static", static_search)

    def static_search_one_axis():
        model = ta.StaticTAM(formula="load ~ s(temperature, k='knots') + l(load_d1)", group_col=GROUP, date_col=DATE)
        best = model.grid_search_fit(fit, validation, {"knots": [4, 6, 10]})
        res.forecast("static_one_axis", test[TARGET], best.predict(test)[f"Estimated{TARGET}"])

    res.attempt("static_one_axis", static_search_one_axis)

    def adaptive_search():
        df = frame()
        model = ta.AdaptiveTAM(adaptive_formula="load ~ s(temperature, k='knots') + l(load_d1)", update_interval_periods=14, training_window_periods=56,
                               steps_per_period=1, group_col=GROUP, date_col=DATE)
        best = model.grid_search_fit(df[df[DATE] < pd.Timestamp("2023-08-01")], {"knots": [4, 8]})
        out = best.predict_online(df[df[DATE] < pd.Timestamp("2023-08-01")])
        part = out[out[DATE] >= pd.Timestamp("2023-06-01")]
        res.forecast("adaptive", part[TARGET], part[f"Estimated{TARGET}"])

    res.attempt("adaptive", adaptive_search)
