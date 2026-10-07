# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Penalties chosen by generalised cross-validation: the gamma of the criterion, the search bounds and steps, a tensor product, the summary table."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

FORMULA = "load ~ s(temperature, k=10) + s(toy, k=10) + " + BASE


def run(res):
    for gamma in (1.0, 1.4, 2.0):
        res.static(f"gamma{gamma:g}", FORMULA, auto=True, fit_args={"gamma": gamma, "number_of_steps": 6})
    res.static("bounds_narrow", FORMULA, auto=True, fit_args={"alpha_p_bounds": (-12.0, 0.0), "number_of_steps": 6})
    res.static("steps12", FORMULA, auto=True, fit_args={"number_of_steps": 12})
    res.static("tensor", "load ~ te(s(temperature, k=6), s(toy, k=6)) + " + BASE, auto=True, fit_args={"number_of_steps": 6})
    res.static("categorical_and_linear", "load ~ l(temperature) + c(month, n_cat=12, topo='nominal') + " + BASE, auto=True, fit_args={"number_of_steps": 6})

    def summary():
        model = res.static("summary_model", FORMULA, auto=True, fit_args={"number_of_steps": 6})
        table = model.summary()
        res.value("summary.columns", ", ".join(table.columns))
        res.value("summary.rows", len(table))
        # the penalty of the offset hardly changes the fit: the criterion is flat above 10^3 and the search stops at a point that depends on the machine, so it is capped
        res.value("summary.regularisation", ", ".join(str(min(float(v), 3.0)) for v in table[table.columns[-1]]))

    res.attempt("summary", summary)

    def offset_penalty():
        train, test = res.train, res.test
        formula = "load ~ s(temperature, k=10, ap=-3) + s(toy, k=10, ap=-9) + " + BASE
        rmse = {}
        for penalty in (3.0, 6.0, 12.0):
            model = ta.StaticTAM(formula=formula, group_col=GROUP, date_col=DATE, default_alpha_p=penalty).fit(train)
            error = test[TARGET].to_numpy() - model.predict(test)[f"Estimated{TARGET}"].to_numpy()
            rmse[penalty] = float(np.sqrt(np.mean(error ** 2)))
        res.gap("offset_penalty.rmse_gap", max(rmse.values()) - min(rmse.values()), tolerance=1e-3)

    res.attempt("offset_penalty", offset_penalty)
