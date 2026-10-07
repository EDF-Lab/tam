# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The prediction API: forecasts on any row order, on a subset of the groups, with extra columns; the decomposition into effects, its names and its sum."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

FORMULA = "load ~ s(temperature, k=8) + l(load_d7) + te(s(temperature, k=5), s(toy, k=5)) + " + BASE


def run(res):
    holder = {}

    def forecast_rows():
        model = holder["model"] = res.static("reference", FORMULA)
        test = res.test
        reference = model.predict(test)[f"Estimated{TARGET}"].to_numpy()
        shuffled = test.sample(frac=1.0, random_state=0)
        out = model.predict(shuffled)
        res.value("order.shuffled_same_forecast_per_row", bool(np.allclose(out[f"Estimated{TARGET}"].to_numpy(), reference[shuffled.index.to_numpy()], rtol=1e-10)))
        duplicated = test.copy()
        duplicated.index = np.zeros(len(duplicated), dtype=int)
        res.value("order.duplicate_labels_same_forecast", bool(np.allclose(model.predict(duplicated)[f"Estimated{TARGET}"].to_numpy(), reference, rtol=1e-10)))
        extra = test.assign(unused_column=1.0, another="x")
        res.value("order.extra_columns_same_forecast", bool(np.allclose(model.predict(extra)[f"Estimated{TARGET}"].to_numpy(), reference, rtol=1e-10)))
        one = test[test[GROUP] == 5]
        res.forecast("subset.one_group", one[TARGET], model.predict(one)[f"Estimated{TARGET}"])
        res.value("subset.rows", len(model.predict(one)))
        res.value("columns", ", ".join(model.predict(test).columns[-3:]))

    res.attempt("rows", forecast_rows)

    def decomposition():
        model, test = holder["model"], res.test
        parts = model.decompose_prediction(test)
        effects = [c for c in parts.columns if c.startswith("effect_")]
        res.value("decompose.effects", ", ".join(effects))
        res.value("decompose.rows", len(parts))
        total = parts[effects].sum(axis=1).to_numpy()
        res.gap("decompose.max_gap_to_forecast", np.max(np.abs(total - model.predict(test)[f"Estimated{TARGET}"].to_numpy())))
        for column in effects:
            res.forecast(f"decompose.{column}", np.zeros(len(parts)), parts[column])

    res.attempt("decomposition", decomposition)

    def summary():
        table = holder["model"].summary()
        res.value("summary.columns", ", ".join(table.columns))
        res.value("summary.features", ", ".join(str(v) for v in table[table.columns[0]]))
        res.value("summary.complexity", ", ".join(str(v) for v in table[table.columns[2]]))

    res.attempt("summary", summary)

    def refit_is_stable():
        again = ta.StaticTAM(formula=FORMULA, group_col=GROUP, date_col=DATE).fit(res.train)
        res.value("refit.same_forecast", bool(np.array_equal(again.predict(res.test)[f"Estimated{TARGET}"].to_numpy(), holder["model"].predict(res.test)[f"Estimated{TARGET}"].to_numpy())))

    res.attempt("refit", refit_is_stable)

    def one_series():
        train, test = res.train[res.train[GROUP] == 12], res.test[res.test[GROUP] == 12]
        model = ta.StaticTAM(formula="load ~ s(temperature, k=8) + l(load_d1)", date_col=DATE).fit(train)
        res.forecast("no_group_column", test[TARGET], model.predict(test)[f"Estimated{TARGET}"])

    res.attempt("no_group_column", one_series)

    def fit_on_a_subset_of_hours():
        train = res.train[res.train[GROUP] < 6]
        model = ta.StaticTAM(formula="load ~ s(temperature, k=8) + l(load_d1)", group_col=GROUP, date_col=DATE).fit(train)
        test = res.test[res.test[GROUP] < 6]
        res.forecast("six_groups", test[TARGET], model.predict(test)[f"Estimated{TARGET}"])

    res.attempt("six_groups", fit_on_a_subset_of_hours)
