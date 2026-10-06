# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conformal intervals: split conformal on a point model, studentised on a distributional model, Mondrian strata, online adaptation (ACI)."""
import numpy as np

import tam as ta
from common import BASE, DATE, GROUP, TARGET

POINT = "load ~ s(temperature, k=8) + " + BASE
MU = "load ~ s(temperature, k=8) + " + BASE
SIGMA = "~ s(temperature, k=6) + c(day_type_week, n_cat=7, topo='nominal')"


def coverage(res, name, frame, table, lower="Lower", upper="Upper"):
    inside = (frame[TARGET].to_numpy() >= table[lower].to_numpy()) & (frame[TARGET].to_numpy() <= table[upper].to_numpy())
    res.value(f"{name}.coverage", float(inside.mean()))
    res.value(f"{name}.mean_width", float((table[upper] - table[lower]).mean()))


def run(res):
    fit = res.train                                              # fitted on the training days; the held-out days are split in two, alternately:
    half = (res.test[DATE].dt.dayofyear // 5) % 2                # one half calibrates the intervals, the other one checks them
    calibration, test = res.test[half == 0].reset_index(drop=True), res.test[half == 1].reset_index(drop=True)

    def split_conformal():
        model = ta.StaticTAM(formula=POINT, group_col=GROUP, date_col=DATE).fit(fit)
        for alpha in (0.1, 0.2):
            model.calibrate_conformal(calibration, alpha=alpha)
            table = model.predict_intervals(test)
            coverage(res, f"split_alpha{alpha:g}", test, table)
            res.value(f"split_alpha{alpha:g}.alpha_t", float(table["Alpha_t"].iloc[0]))
        model.calibrate_conformal(calibration, alpha=0.1)
        online = model.predict_intervals(test.sort_values(DATE), method="aci", gamma=0.05, y_true_online=test.sort_values(DATE)[TARGET].to_numpy())
        coverage(res, "aci", test.sort_values(DATE), online)
        res.value("aci.final_alpha_t", float(online["Alpha_t"].iloc[-1]))

    res.attempt("split_conformal", split_conformal)

    def studentised():
        model = ta.StaticTAM(formula={"mu": MU, "sigma": SIGMA}, group_col=GROUP, date_col=DATE, dist_kwargs={"log_target": False}).fit(fit)
        model.calibrate_conformal(calibration, alpha=0.1, studentized=True)
        coverage(res, "studentised", test, model.predict_intervals(test))

    res.attempt("studentised", studentised)

    def mondrian():
        model = ta.StaticTAM(formula={"mu": MU, "sigma": SIGMA}, group_col=GROUP, date_col=DATE, dist_kwargs={"log_target": False}).fit(fit)
        conformal = ta.ConformalDistributionalTAM(model, alpha=0.1, strata_col="day_type_jf").calibrate(calibration)
        table = conformal.predict_interval(test)
        coverage(res, "mondrian", test, table, "lower", "upper")
        res.forecast("mondrian.pvalue", np.zeros(len(test)), conformal.conformal_pvalue(test))
        flags = conformal.anomaly(test)
        res.value("mondrian.anomalies", int(flags["is_anomaly"].sum()))
        ordered = test.sort_values(DATE)
        coverage(res, "mondrian_aci", ordered, conformal.aci_intervals(ordered, gamma=0.02), "lower", "upper")

    res.attempt("mondrian", mondrian)

    def engine():
        rng = np.random.default_rng(0)
        engine = ta.SafetyTAM(alpha=0.1).calibrate_scores(np.abs(rng.normal(size=500)))
        res.value("engine.quantile_alpha0.1", engine.conformal_quantile())
        res.value("engine.quantile_alpha0.3", engine.conformal_quantile(0.3))
        res.forecast("engine.pvalue", np.zeros(5), engine.pvalue(np.array([0.1, 0.5, 1.0, 2.0, 4.0])))

    res.attempt("engine", engine)
