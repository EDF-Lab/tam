# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Penalties chosen by generalised cross-validation: the gamma of the criterion, the search bounds and steps, a tensor product, the summary table."""
from common import BASE

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
        res.value("summary.regularisation", ", ".join(str(v) for v in table[table.columns[-1]]))

    res.attempt("summary", summary)
