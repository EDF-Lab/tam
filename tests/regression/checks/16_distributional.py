# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Distributional (location-scale) models: quantiles, median, CDF, CRPS, density, anomaly score, the tail family."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

MU = "load ~ s(temperature, k=8) + " + BASE
SIGMA = "~ s(temperature, k=6) + c(day_type_week, n_cat=7, topo='nominal')"


def fit(res, **dist_kwargs):
    return ta.StaticTAM(formula={"mu": MU, "sigma": SIGMA}, group_col=GROUP, date_col=DATE, dist_kwargs=dist_kwargs).fit(res.train)


def run(res):
    holder = {}

    def location_scale():
        model = holder["model"] = fit(res, log_target=False)
        test = res.test
        quantiles = model.predict_quantiles(test, taus=(0.05, 0.25, 0.5, 0.75, 0.95))
        for column in quantiles.columns:
            if column.startswith("q"):
                res.forecast(f"quantile.{column}", test[TARGET], quantiles[column])
        res.forecast("median", test[TARGET], model.predict_median(test))
        res.forecast("cdf", np.full(len(test), 0.5), model.cdf(test))
        res.forecast("crps", np.zeros(len(test)), model.crps(test))
        res.value("tail_family", getattr(model, "tail_family_", "none"))

    res.attempt("location_scale", location_scale)

    def anomaly():
        table = holder["model"].anomaly_score(res.test)
        for column in ("predicted_median", "z_score", "tail_pvalue", "anomaly_score"):
            res.forecast(f"anomaly.{column}", np.zeros(len(table)), table[column])
        for side, count in table["side"].value_counts().sort_index().items():
            res.value(f"anomaly.side_{side}", count)

    res.attempt("anomaly_score", anomaly)

    def tails():
        for family in ("normal", "student_t", "auto"):
            model = fit(res, log_target=False, tail_family=family)
            q = model.predict_quantiles(res.test, taus=(0.01, 0.99))
            res.forecast(f"tail_{family}.upper", res.test[TARGET], q[[c for c in q.columns if c.startswith("q")][-1]])

    res.attempt("tails", tails)

    def log_target():
        model = fit(res, log_target=True)
        res.forecast("log_target.median", res.test[TARGET], model.predict_median(res.test))

    res.attempt("log_target", log_target)

    def mixture():
        train, test = res.train[res.train[GROUP] == 12], res.test[res.test[GROUP] == 12]         # a mixture is fitted for one series
        model = ta.StaticTAM(formula="load ~ s(temperature, k=8) + l(load_d1)", date_col=DATE, mixture_components=2).fit(train)
        res.forecast("mixture.mean", test[TARGET], model.predict_mean(test))
        res.forecast("mixture.component_means", np.zeros(model.component_means(test).size), model.component_means(test))
        res.forecast("mixture.log_density", np.zeros(len(test)), model.log_density(test))
        res.forecast("mixture.responsibilities", np.zeros(model.responsibilities(test).size), model.responsibilities(test))

    res.attempt("mixture", mixture)
